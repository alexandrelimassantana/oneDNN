/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_RV64_RVJIT_RVJIT_REGISTER_POOL_HPP
#define CPU_RV64_RVJIT_RVJIT_REGISTER_POOL_HPP

#include <array>
#include <cstdint>
#include <initializer_list>

#include "cpu/rv64/rvjit/rvjit_emitter.hpp"
#include "cpu/rv64/rvjit/rvjit_utils.hpp"

#if defined(DNNL_DEV_MODE)
#include "common/verbose.hpp"
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Non-owning 2D view into a contiguous slice of a partition_t or vector_ctx_t
template <typename T>
struct block_t {
    block_t() = default;
    block_t(const T *data, int cols) : data_(data), rows_(1), cols_(cols) {}
    block_t(const T *data, int rows, int cols)
        : data_(data), rows_(rows), cols_(cols) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }

    T operator()(int i) const { return data_[i % cols_]; }
    T operator()(int row, int col) const {
        return data_[(row % rows_) * cols_ + (col % cols_)];
    }
    T operator[](int i) const { return data_[i % size()]; }

    const T *begin() const { return data_; }
    const T *end() const { return data_ + size(); }

    block_t reshaped(int new_rows, int new_cols) const {
        return block_t(data_, new_rows, new_cols);
    }

private:
    const T *data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
};

using x_block_t = block_t<Reg>;
using f_block_t = block_t<FReg>;
#if XBYAK_RISCV_V
using v_block_t = block_t<VReg>;
#endif

/// Fixed-capacity register array with a forward-advancing allocation cursor
template <typename T, int N = 32>
struct partition_t {
    partition_t() = default;

    template <typename It>
    partition_t(It s, It e) : count_(static_cast<int>(e - s)) {
        int i = 0;
        for (auto it = s; it != e; ++it)
            ids_[i++] = *it;
    }

    partition_t(std::initializer_list<T> items)
        : partition_t(items.begin(), items.end()) {}

    int size() const { return count_; }
    int allocated() const { return alloc_; }

    /// Enumerates the full register list regardless of allocation state.
    const T *begin() const { return ids_.data(); }
    const T *end() const { return ids_.data() + count_; }

    /// Allocates a single register
    T allocate() {
        if (alloc_ >= count_) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed partition_t::allocate() due to "
                        "insufficient registers\n");
            });
            return T();
        }
        return ids_[alloc_++];
    }

    /// Allocates a 1D block of registers
    block_t<T> allocate(int cols) { return allocate(1, cols); }

    /// Allocates a 2D block of registers
    ///
    /// @pre Caller must ensure the partition has enough registers
    block_t<T> allocate(int rows, int cols) {
        if (alloc_ + rows * cols > count_) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed partition_t::allocate() due to "
                        "insufficient registers\n");
            });
            return {};
        }
        const T *start = ids_.data() + alloc_;
        alloc_ += rows * cols;
        return block_t<T>(start, rows, cols);
    }

private:
    std::array<T, N> ids_;
    int count_ = 0;
    int alloc_ = 0;
};

#if XBYAK_RISCV_V

/// Vector register allocation context with physical-overlap cross-invalidation
///
/// Maintains an accumulator candidate list and an input candidate list that
/// share a single physical-register occupancy bitmask (one bit per v0..v31).
/// Any allocation from either list marks the consumed physical registers so
/// that overlapping groups in the other list are automatically blocked.
///
/// Example (inp=m4, acc=m8):
///   acc candidates: {v0, v8, v16, v24}   acc_phys_ = 8
///   inp candidates: {v0, v4, ..., v28}    inp_phys_ = 4
///   Allocate acc v0 (bits 0-7):  inp v0 (bits 0-3) and v4 (bits 4-7) blocked
///   Allocate inp v8 (bits 8-11): acc v8 (bits 8-15) blocked
struct vector_ctx_t {
    static constexpr int NVPR = 32;

    VReg acc_cands_[NVPR];
    int acc_n_ = 0;
    int acc_phys_ = 0; // vgroup_size(acc_lmul) — physical registers per group

    VReg inp_cands_[NVPR];
    int inp_n_ = 0;
    int inp_phys_ = 0; // vgroup_size(inp_lmul)

    // Stable storage for block_t pointers into each pool
    VReg acc_out_[NVPR];
    int acc_out_n_ = 0;
    VReg inp_out_[NVPR];
    int inp_out_n_ = 0;

    uint32_t phys_used_ = 0; // bit b set ↔ physical register vb is occupied
    bool has_acc_ = false; // false: single-pool mode, acc falls back to inp

    VReg alloc_inp() {
        const uint32_t bits = (1u << inp_phys_) - 1u;
        for (int i = 0; i < inp_n_; ++i) {
            const uint32_t m = bits << inp_cands_[i].getIdx();
            if (!(phys_used_ & m)) {
                phys_used_ |= m;
                return inp_cands_[i];
            }
        }
        DEBUG({
            verbose_printf(verbose_t::debuginfo,
                    "Failed vector_ctx_t: new_vector() out of inp registers\n");
        });
        return VReg {};
    }

    VReg alloc_acc() {
        const uint32_t bits = (1u << acc_phys_) - 1u;
        for (int i = 0; i < acc_n_; ++i) {
            const uint32_t m = bits << acc_cands_[i].getIdx();
            if (!(phys_used_ & m)) {
                phys_used_ |= m;
                return acc_cands_[i];
            }
        }
        DEBUG({
            verbose_printf(verbose_t::debuginfo,
                    "Failed vector_ctx_t: new_vector_accumulator() out of acc "
                    "registers\n");
        });
        return VReg {};
    }

    VReg alloc_acc_or_inp() { return has_acc_ ? alloc_acc() : alloc_inp(); }

    v_block_t alloc_inp_block(int rows, int cols) {
        return alloc_block(inp_cands_, inp_n_, inp_phys_, inp_out_, inp_out_n_,
                rows, cols, "new_vector()");
    }

    v_block_t alloc_acc_block(int rows, int cols) {
        return alloc_block(acc_cands_, acc_n_, acc_phys_, acc_out_, acc_out_n_,
                rows, cols, "new_vector_accumulator()");
    }

    v_block_t alloc_acc_or_inp_block(int rows, int cols) {
        return has_acc_ ? alloc_acc_block(rows, cols)
                        : alloc_inp_block(rows, cols);
    }

private:
    v_block_t alloc_block(VReg *cands, int n, int phys, VReg *out, int &out_n,
            int rows, int cols, const char *caller) {
        const int start = out_n;
        const uint32_t phys_save = phys_used_;
        const uint32_t bits = (1u << phys) - 1u;
        for (int k = 0; k < rows * cols; ++k) {
            bool found = false;
            for (int i = 0; i < n; ++i) {
                const uint32_t m = bits << cands[i].getIdx();
                if (!(phys_used_ & m)) {
                    phys_used_ |= m;
                    out[out_n++] = cands[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                phys_used_ = phys_save;
                out_n = start;
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "Failed vector_ctx_t::%s block allocation\n",
                            caller);
                });
                return {};
            }
        }
        return v_block_t(out + start, rows, cols);
    }
};

#endif // XBYAK_RISCV_V

/// Component for managing integer, fp, and (with RVV) vector register allocation
struct register_pool_t {

    register_pool_t(emitter_t e) : e_(e) {}

    // Integer register file

    static constexpr int NGPR = 27; // integer registers available (t+a+s)
    using x_partition_t = partition_t<Reg, NGPR>;

    /// Sets the integer register file, optionally excluding the listed registers
    void int_register_file(std::initializer_list<Reg> excl = {}) {
        const Reg *regs = integer_registers();
        x_ctx_ = filter<Reg, NGPR>(regs, NGPR, excl.begin(), excl.end());
        callee_saved_boundary_ = find_callee_saved_boundary(x_ctx_);
    }

    template <typename It>
    void int_register_file_excluding(It s, It e) {
        const Reg *regs = integer_registers();
        x_ctx_ = filter<Reg, NGPR>(regs, NGPR, s, e);
        callee_saved_boundary_ = find_callee_saved_boundary(x_ctx_);
    }

    void int_register_file_excluding(std::initializer_list<Reg> excl) {
        int_register_file_excluding(excl.begin(), excl.end());
    }

    Reg new_int() { return x_ctx_.allocate(); }
    x_block_t new_int(int cols) { return x_ctx_.allocate(cols); }
    x_block_t new_int(int r, int c) { return x_ctx_.allocate(r, c); }

    // Float register file

    static constexpr int NFPR = 20; // caller-saved fp registers available
    using f_partition_t = partition_t<FReg, NFPR>;

    /// Sets the float register file, optionally excluding the listed registers
    void float_register_file(std::initializer_list<FReg> excl = {}) {
        const FReg *regs = float_registers();
        f_ctx_ = filter<FReg, NFPR>(regs, NFPR, excl.begin(), excl.end());
    }

    void float_register_file_excluding(std::initializer_list<FReg> excl) {
        float_register_file(excl);
    }

    FReg new_float() { return f_ctx_.allocate(); }
    f_block_t new_float(int cols) { return f_ctx_.allocate(cols); }
    f_block_t new_float(int r, int c) { return f_ctx_.allocate(r, c); }

    // Const helper

    /// Conditionally allocates a register for a value that !is_simm12()
    const_t new_const(int value) {
        return is_simm12(value) ? const_t(value) : const_t(new_int());
    }

    // Callee-saved save / restore

    /// Preserves all callee-saved gpr allocated using this component
    ///
    /// @pre Caller must not further allocate registers after calling preserve
    void preserve() {
        const int hi = x_ctx_.allocated();
        if (hi <= callee_saved_boundary_) return;
        const int lo = callee_saved_boundary_;
        const int n = hi - lo;
        const Reg *ids = x_ctx_.begin();
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -(n * 8));
        for (int i = 0; i < n; ++i)
            e_->sd(ids[lo + i], Xbyak_riscv::sp, i * 8);
    }

    /// Restores all callee-saved gpr allocated using this component
    ///
    /// @pre Caller must not further allocate registers after calling preserve
    void restore() {
        const int hi = x_ctx_.allocated();
        if (hi <= callee_saved_boundary_) return;
        const int lo = callee_saved_boundary_;
        const int n = hi - lo;
        const Reg *ids = x_ctx_.begin();
        for (int i = 0; i < n; ++i)
            e_->ld(ids[lo + i], Xbyak_riscv::sp, i * 8);
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, n * 8);
    }

    /// Preserves a list of gpr in the same order as they are given
    void preserve(std::initializer_list<Reg> list) {
        const int n = list.size();
        if (n == 0) return;
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -(n * 8));
        for (int i = 0; i < n; ++i)
            e_->sd(*(list.begin() + i), Xbyak_riscv::sp, i * 8);
    }

    /// Restores a list of gpr in the same order as they are given
    void restore(std::initializer_list<Reg> list) {
        const int n = list.size();
        if (n == 0) return;
        for (int i = 0; i < n; ++i)
            e_->ld(*(list.begin() + i), Xbyak_riscv::sp, i * 8);
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, n * 8);
    }

#if XBYAK_RISCV_V

    /// Sets a single vector register file (non-widening kernels)
    void vector_register_file(
            const LMUL &m, std::initializer_list<VReg> excl = {}) {
        v_ctx_ = {};
        const uint32_t em = excl_vmask(excl.begin(), excl.end());
        int count = 0;
        const VReg *regs = vector_registers(m, count);
        for (int i = 0; i < count; ++i)
            if (!(em & (1u << regs[i].getIdx())))
                v_ctx_.inp_cands_[v_ctx_.inp_n_++] = regs[i];
        v_ctx_.inp_phys_ = vgroup_size(m);
    }

    /// Sets two vector register files for widening kernels
    ///
    /// Both lists share a physical-register occupancy bitmask so that
    /// allocating from one automatically cross-invalidates overlapping
    /// candidates in the other.
    ///
    /// @param inp  LMUL for input (narrow) operands
    /// @param acc  LMUL for accumulator (wide) operands; must be >= inp
    void vector_register_file(const LMUL &inp, const LMUL &acc,
            std::initializer_list<VReg> excl = {}) {
        if (inp == acc) {
            vector_register_file(inp, excl);
            return;
        }
        v_ctx_ = {};
        const uint32_t em = excl_vmask(excl.begin(), excl.end());

        int inp_count = 0;
        const VReg *inp_regs = vector_registers(inp, inp_count);
        for (int i = 0; i < inp_count; ++i)
            if (!(em & (1u << inp_regs[i].getIdx())))
                v_ctx_.inp_cands_[v_ctx_.inp_n_++] = inp_regs[i];
        v_ctx_.inp_phys_ = vgroup_size(inp);

        int acc_count = 0;
        const VReg *acc_regs = vector_registers(acc, acc_count);
        for (int i = 0; i < acc_count; ++i)
            if (!(em & (1u << acc_regs[i].getIdx())))
                v_ctx_.acc_cands_[v_ctx_.acc_n_++] = acc_regs[i];
        v_ctx_.acc_phys_ = vgroup_size(acc);

        v_ctx_.has_acc_ = true;
    }

    VReg new_vector() { return v_ctx_.alloc_inp(); }
    v_block_t new_vector(int cols) { return v_ctx_.alloc_inp_block(1, cols); }
    v_block_t new_vector(int rows, int cols) {
        return v_ctx_.alloc_inp_block(rows, cols);
    }

    /// Allocates accumulator registers from the acc pool (or inp if single-pool)
    ///
    /// @note In single-pool mode (vector_register_file(LMUL)) the accumulator
    ///       and input pools are the same; callers are responsible for ensuring
    ///       the allocated register names are valid for the effective acc LMUL.
    VReg new_vector_accumulator() { return v_ctx_.alloc_acc_or_inp(); }
    v_block_t new_vector_accumulator(int cols) {
        return v_ctx_.alloc_acc_or_inp_block(1, cols);
    }
    v_block_t new_vector_accumulator(int rows, int cols) {
        return v_ctx_.alloc_acc_or_inp_block(rows, cols);
    }

#endif

private:
    emitter_t e_;
    x_partition_t x_ctx_;
    f_partition_t f_ctx_;
    int callee_saved_boundary_ = 0;
#if XBYAK_RISCV_V
    vector_ctx_t v_ctx_;
#endif

    // Scans `p` forward to find the first index containing an s-register
    static int find_callee_saved_boundary(const x_partition_t &p) {
        // s0=x8, s1=x9, s2..s11=x18..x27
        static constexpr uint32_t s_mask = (1u << 8) | (1u << 9) | (1u << 18)
                | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) | (1u << 23)
                | (1u << 24) | (1u << 25) | (1u << 26) | (1u << 27);
        const Reg *ids = p.begin();
        for (int i = 0; i < p.size(); ++i)
            if (s_mask & (1u << ids[i].getIdx())) return i;
        return p.size();
    }

    static const Reg *integer_registers() {
        using namespace Xbyak_riscv;
        static const Reg x[NGPR]
                = {t0, t1, t2, t3, t4, t5, t6, a0, a1, a2, a3, a4, a5, a6, a7,
                        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
        return x;
    }

    static const FReg *float_registers() {
        using namespace Xbyak_riscv;
        static const FReg f[NFPR] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7,
                ft8, ft9, ft10, ft11, fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7};
        return f;
    }

    template <typename T, int N, typename It>
    static partition_t<T, N> filter(const T *list, int list_size, It s, It e) {
        uint32_t excl_mask = 0;
        for (auto it = s; it != e; ++it)
            excl_mask |= 1u << (*it).getIdx();
        T buf[N];
        int n = 0;
        for (int i = 0; i < list_size; ++i)
            if (!(excl_mask & (1u << list[i].getIdx()))) buf[n++] = list[i];
        return partition_t<T, N>(buf, buf + n);
    }

#if XBYAK_RISCV_V
    template <typename It>
    static uint32_t excl_vmask(It s, It e) {
        uint32_t m = 0;
        for (auto it = s; it != e; ++it)
            m |= 1u << (*it).getIdx();
        return m;
    }

    static const VReg *vector_registers(const LMUL &m, int &count) {
        using namespace Xbyak_riscv;
        switch (m) {
            case LMUL::m8: {
                static const VReg regs[] = {v0, v8, v16, v24};
                count = 4;
                return regs;
            }
            case LMUL::m4: {
                static const VReg regs[]
                        = {v0, v4, v8, v12, v16, v20, v24, v28};
                count = 8;
                return regs;
            }
            case LMUL::m2: {
                static const VReg regs[] = {v0, v2, v4, v6, v8, v10, v12, v14,
                        v16, v18, v20, v22, v24, v26, v28, v30};
                count = 16;
                return regs;
            }
            default: {
                static const VReg regs[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8,
                        v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
                        v31};
                count = 32;
                return regs;
            }
        }
    }
#endif
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
