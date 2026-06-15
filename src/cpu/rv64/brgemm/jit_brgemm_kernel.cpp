/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include "cpu/rv64/brgemm/jit_brgemm_kernel.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

// Single generator for uniform and widening operations
struct jit_brgemm_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    jit_brgemm_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1

    static constexpr int N_UR = 4;
    static constexpr int K_UR = 4;

    const data_type_t dt_a = brg_.dt_a;
    const data_type_t dt_c = brg_.dt_c;

    const int sewba = brg_.typesize_A;
    const int sewbb = brg_.typesize_B;
    const int sewbc = brg_.typesize_C;
    const dim_t LDA_bytes = brg_.LDA * sewba;
    const dim_t LDB_bytes = brg_.LDB * sewbb;
    const dim_t LDC_bytes = brg_.LDC * sewbc;

    // Setup rvjit component system
    rvjit_t m(*this);
    auto &ca = m.const_arithmetic();
    auto &cf = m.control_flow();
    auto &pool = m.register_pool();
    auto &mem = m.memory_move();
    auto &mat = m.matmul();

    // Live registers
    const Reg args = a0; // Arguments
    const Reg ptra = a1; // A pointer (advances per K block)
    const Reg ptrb = a2; // B pointer (advances per N block)
    const Reg ptrc = a3; // C pointer (advances per N block)
    const Reg N = a4; // N loop size (dynamic)
    const Reg n = a5; // N loop iter
    const Reg K = a6; // K loop size (dynamic)
    const Reg k = a7; // K loop iter

    const LMUL lm = LMUL::m4;
    const SEW sew_c = sew_for(dt_c);

    // Integer and float register files — vector file is set up inside configure
    pool.int_register_file_excluding({args, ptra, ptrb, ptrc, N, n, K, k});
    pool.float_register_file();

    // Stride constants for A and C; B outer stride depends on the simm12 check
    const const_t lda = pool.new_const(LDA_bytes);
    // Use a single shared B pointer when the whole [K_UR, N_UR] tile offset fits
    // simm12; otherwise allocate one pointer per column.
    const bool one_ptrb = is_simm12(N_UR * LDB_bytes + K_UR * sewbb);
    const const_t b_outer
            = one_ptrb ? const_t(LDB_bytes) : const_t(pool.new_int());
    const const_t ldc = pool.new_const(LDC_bytes);

    // configure allocates c_data, a_data, b_data, pivots and sets up the vector file
    mat.configure(N_UR, dt_a, dt_c, lm, ptra, lda,
            const_t(0), // A: K-step advance = lda, unit vle
            ptrb, b_outer, const_t(sewbb), // B: column stride, K-step stride
            ptrc);

    const x_block_t tmp = pool.new_int(2);
    const Reg avl = tmp[0]; // Used once at the start of the kernel
    const Reg Ntmp = tmp[0]; // Needed during the N loop
    const Reg Ktmp = tmp[1]; // Needed during the K loop
    const Reg bias = tmp[1]; // bias pointer during post-ops
    const Reg beta = tmp[1]; // beta bits during post-ops
    const Reg Boff = tmp[1]; // B matrix offset temporary, if needed

    const VReg vtmp = mat.scratch_vreg(); // free after k_loop completes

    // Code start

    pool.preserve();

    // Load arguments
    ld(ptra, args, offsetof(brgemm_kernel_params_t, ptr_A));
    ld(ptrb, args, offsetof(brgemm_kernel_params_t, ptr_B));
    ld(ptrc, args, offsetof(brgemm_kernel_params_t, ptr_C));
    ld(N, args, offsetof(brgemm_kernel_params_t, N));
    ld(K, args, offsetof(brgemm_kernel_params_t, K));

    // Initialize stride constants that didn't fit simm12
    if (lda.is_reg()) ca.init_constant(LDA_bytes, lda.reg());
    if (b_outer.is_reg()) ca.init_constant(LDB_bytes, b_outer.reg());
    if (ldc.is_reg()) ca.init_constant(LDC_bytes, ldc.reg());

    // Configure vector type (wide/accumulator side)
    ld(avl, args, offsetof(brgemm_kernel_params_t, M));
    vsetvli(x0, avl, sew_c, lm, VTA::ta, VMA::ma);

    // N loop unrolled by {N_UR, 1}
    const auto plan = dispatch_plan_t::main_then_dispatch(N_UR, n, N, Ntmp);
    cf.dispatch(plan, [&](int nb) {
        mat.dense_loop(nb, K_UR, k, K, Ktmp);

        const v_block_t c = mat.c_data();

        // Apply bias conditionally
        ld(bias, args, offsetof(brgemm_kernel_params_t, ptr_bias));
        cf.if_nez(bias, [&] {
            mem.vle(vtmp, bias, dt_c);
            for (int c_idx = 0; c_idx < nb; ++c_idx)
                vfadd_vv(c[c_idx], c[c_idx], vtmp);
        });

        // Apply beta conditionally
        lw(beta, args, offsetof(brgemm_kernel_params_t, beta));
        cf.if_nez(beta, [&](bool nez) {
            if (nez) {
                // beta != 0: C[col] = C[col] + accum
                for (int c_idx = 0; c_idx < nb; ++c_idx) {
                    mem.vle(vtmp, ptrc, dt_c);
                    vfadd_vv(vtmp, vtmp, c[c_idx]);
                    mem.vse(vtmp, ptrc, dt_c);
                    ca.add_const(ptrc, ptrc, ldc);
                }
            } else {
                // beta == 0: C[col] = accum (overwrite)
                for (int c_idx = 0; c_idx < nb; ++c_idx) {
                    mem.vse(c[c_idx], ptrc, dt_c);
                    ca.add_const(ptrc, ptrc, ldc);
                }
            }
        });

        // Advance B pointer for next N group
        const const_t B_off = ca.init_constant(nb * LDB_bytes, Boff);
        ca.add_const(ptrb, ptrb, B_off);
    });

    pool.restore();
    ret();
#else
    // RVV JIT is disabled at build time.
    ret();
#endif
}

brgemm_kernel_common_t::brgemm_kernel_common_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_kernel_t(brg)) {}

brgemm_kernel_common_t::~brgemm_kernel_common_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_common_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_common_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
