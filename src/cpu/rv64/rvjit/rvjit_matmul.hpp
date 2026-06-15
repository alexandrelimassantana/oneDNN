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

#ifndef CPU_RV64_RVJIT_RVJIT_MATMUL_HPP
#define CPU_RV64_RVJIT_RVJIT_MATMUL_HPP

#include "common/c_types_map.hpp"

#include "cpu/rv64/rvjit/rvjit_const_arithmetic.hpp"
#include "cpu/rv64/rvjit/rvjit_control_flow.hpp"
#include "cpu/rv64/rvjit/rvjit_emitter.hpp"
#include "cpu/rv64/rvjit/rvjit_fma.hpp"
#include "cpu/rv64/rvjit/rvjit_memory_move.hpp"
#include "cpu/rv64/rvjit/rvjit_register_pool.hpp"
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

#if XBYAK_RISCV_V

/// Pipelined vector x scalar matmul-accumulate component for RVV kernels
///
/// @note Caller must set vsetvli to match dt_inp/lmul before dense_loop
class rvv_matmul_t {
public:
    explicit rvv_matmul_t(emitter_t e, register_pool_t &pool,
            memory_move_t &mem, rvv_macc_emitter_t &mc, const_arithmetic_t &ca,
            control_flow_t &cf)
        : e_(e), pool_(pool), mem_(mem), mc_(mc), ca_(ca), cf_(cf) {}

    /// Allocates data registers and stores the addressing configuration
    ///
    /// @param N         max N-unroll for register allocation
    /// @param dt_inp    input element type for A and B
    /// @param dt_acc    accumulator element type for C
    /// @param lmul      LMUL for input operands
    /// @param ptra      A matrix base pointer (advances per K step inside dense_loop)
    /// @param a_outer   K-step advance for ptra (bytes; imm or runtime reg)
    /// @param a_inner   within-vector element stride; imm(0) → vle, reg → vlse
    /// @param ptrb      B matrix base pointer (managed by caller between N-tiles)
    /// @param b_outer   B inter-column stride; imm → 1 pivot, reg → N pivots
    /// @param b_inner   B K-step stride; imm → batch advance at end, reg → per-step
    /// @param ptrc      C matrix pointer (exposed via ptrc() for post-ops)
    bool configure(int N, data_type_t dt_inp, data_type_t dt_acc, LMUL lmul,
            Reg ptra, const_t a_outer, const_t a_inner, Reg ptrb,
            const_t b_outer, const_t b_inner, Reg ptrc) {
        N_ = N;
        dt_inp_ = dt_inp;
        ptra_ = ptra;
        ptrb_ = ptrb;
        ptrc_ = ptrc;
        a_outer_stride_ = a_outer;
        a_inner_stride_ = a_inner;
        b_outer_stride_ = b_outer;
        b_inner_stride_ = b_inner;

        op_ = (sewb(dt_inp) == sewb(dt_acc)) ? fma_t::uniform(dt_inp)
                                             : fma_t::widening(dt_inp);
        const LMUL lmul_acc
                = lmul_for(vgroup_size(lmul) * sewb(dt_acc) / sewb(dt_inp));

        pool_.vector_register_file(lmul, lmul_acc);
        c_data_ = pool_.new_vector_accumulator(N, 1);
        a_data_ = pool_.new_vector(2, 1);
        b_data_ = pool_.new_float(2, N);
        pivots_ = pool_.new_int(b_outer.is_reg() ? N : 1);

        if (!c_data_.size()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed matmul::configure() due to "
                        "insufficient accumulator registers\n");
            });
            return false;
        }
        if (!a_data_.size()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed matmul::configure() due to "
                        "insufficient vector registers\n");
            });
            return false;
        }
        if (!b_data_.size()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed matmul::configure() due to "
                        "insufficient float registers\n");
            });
            return false;
        }
        if (!pivots_.size()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed matmul::configure() due to "
                        "insufficient pointer registers\n");
            });
            return false;
        }

        amode_ = a_inner.is_reg() ? vaddr_t::strided(ptra, a_inner.reg())
                                  : vaddr_t::unit(ptra);
        return true;
    }

    /// Emits B pivot init, C zero-init, mv(k,0), and an unrolled K-loop
    ///
    /// @param nb    N-unroll for this tile (≤ N passed to configure)
    /// @param kb    main K-unroll factor; tail always uses 1
    /// @param k     K loop counter register (written: reset to 0 on entry)
    /// @param K     K loop bound register (read-only)
    /// @param Ktmp  scratch register for the unrolled-loop limit computation
    void dense_loop(int nb, int kb, Reg k, Reg K, Reg Ktmp) {
        e_->vmv_v_i(c_data_(0, 0), 0);
        e_->mv(pivots_[0], ptrb_);
        for (int nn = 1; nn < nb; ++nn) {
            e_->vmv_v_i(c_data_(nn, 0), 0);
            if (b_outer_stride_.is_reg())
                for (int i = 1; i < nb; ++i)
                    ca_.add_const(pivots_[i], pivots_[i - 1], b_outer_stride_);
        }

        e_->mv(k, Xbyak_riscv::x0);
        cf_.unrolled_loops(
                k, K, Ktmp, {kb, 1}, [&](int ku) { k_loop_(nb, ku); });
    }

    v_block_t c_data() const { return c_data_; }
    /// Returns a scratch vector register (a_data ring slot 0, free after dense_loop)
    VReg scratch_vreg() const { return a_data_(0, 0); }
    Reg ptrb() const { return ptrb_; }
    Reg ptrc() const { return ptrc_; }

private:
    emitter_t e_;
    register_pool_t &pool_;
    memory_move_t &mem_;
    rvv_macc_emitter_t &mc_;
    const_arithmetic_t &ca_;
    control_flow_t &cf_;

    int N_ = 0;
    data_type_t dt_inp_ = data_type::undef;
    fma_t op_ = fma_t::uniform(data_type::f32);

    Reg ptra_ {}, ptrb_ {}, ptrc_ {};
    const_t a_outer_stride_, a_inner_stride_;
    const_t b_outer_stride_, b_inner_stride_;

    vaddr_t amode_ {};
    v_block_t c_data_, a_data_;
    f_block_t b_data_;
    x_block_t pivots_;

    // Pipelined K-loop body for a single batch of kb steps
    void k_loop_(int nb, int kb) {
        load_a_(0, kb);
        for (int nn = 0; nn < nb; ++nn)
            load_b_(0, kb, nn, nb);

        for (int kk = 0; kk < kb; ++kk) {
            load_a_(kk + 1, kb);
            for (int nn = 0; nn < nb; ++nn) {
                mc_.fmacc_float(
                        c_data_(nn, 0), b_data_(kk, nn), a_data_(kk, 0), op_);
                load_b_(kk + 1, kb, nn, nb);
            }
        }

        // Cases A and C (b_inner is imm): batch pivot advance after the loop
        if (!b_inner_stride_.is_reg()) {
            const int n_piv = b_outer_stride_.is_reg() ? nb : 1;
            for (int i = 0; i < n_piv; ++i)
                ca_.add_const(pivots_[i], pivots_[i],
                        const_t(kb * b_inner_stride_.imm()));
        }
    }

    // Loads A[kk] into ring slot ki%2 and advances ptra; no-op when ki >= kb
    void load_a_(int ki, int kb) {
        if (ki >= kb) return;
        mem_.vload(a_data_(ki, 0), amode_, dt_inp_);
        ca_.add_const(ptra_, ptra_, a_outer_stride_);
    }

    // Loads B[kk][nn] from the appropriate pivot; no-op when kk >= kb
    //
    // Case B: advances the single pivot once per k-step (nn==0, kk>0).
    // kk>0 suppresses the advance during the priming call (load_b_(0,0));
    // loop-body calls always pass kk+1 >= 1 so the advance fires every iteration,
    // including the last (kk+1 >= kb_) where the load is skipped but the pivot
    // must still be left pointing at the next batch's k=0.
    void load_b_(int ki, int kb, int ni, int nb) {
        if (ni == 0 && ki > 0 && b_inner_stride_.is_reg())
            ca_.add_const(pivots_[0], pivots_[0], b_inner_stride_);
        if (ki >= kb) return;
        if (b_outer_stride_.is_reg()) {
            // Case C: N pivots, ki contributes to the immediate
            mem_.fload(b_data_(ki, ni), pivots_[ni], dt_inp_,
                    ki * b_inner_stride_.imm());
        } else if (b_inner_stride_.is_reg()) {
            // Case B: 1 pivot advancing per step, nn contributes to the immediate
            mem_.fload(b_data_(ki, ni), pivots_[0], dt_inp_,
                    ni * b_outer_stride_.imm());
        } else {
            // Case A: 1 pivot, combined compile-time offset
            mem_.fload(b_data_(ki, ni), pivots_[0], dt_inp_,
                    ni * b_outer_stride_.imm() + ki * b_inner_stride_.imm());
        }

        // Pointer updates
        if (ki == kb) {
            // Cases A and C (b_inner is imm): batch pivot advance after the last iteration
            if (!b_inner_stride_.is_reg()) {
                if (b_outer_stride_.is_reg() || ni == 0) {
                    ca_.add_const(pivots_[ni], pivots_[ni],
                            const_t(kb * b_inner_stride_.imm()));
                }
            }
        }
    }
};

#endif // XBYAK_RISCV_V

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
