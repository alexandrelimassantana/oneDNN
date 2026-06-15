/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#include "cpu/rv64/gemm/jit_rvv_gemm_kernel.hpp"
#include "common/verbose.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;
using namespace rvjit;

jit_rvv_gemm_kernel_t::jit_rvv_gemm_kernel_t(
        bool isTransA, bool isTransB, bool has_bias)
    : jit_generator_t("rv64_gemm_kernel_f32_jit")
    , isTransA_(isTransA)
    , isTransB_(isTransB)
    , has_bias_(has_bias) {
    create_kernel();
}

void jit_rvv_gemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1

    // Unrolling patterns
    static constexpr int N_UR = 6;
    static constexpr int K_UR = 4;

    // Operand data types — uniform f32
    const data_type_t dt_a = data_type::f32;
    const data_type_t dt_c = data_type::f32;
    const int sewba = sizeof(float);
    const int sewbb = sizeof(float);
    const int sewbc = sizeof(float);

    // rvjit component system
    rvjit_t m(*this);
    auto &cf = m.control_flow();
    auto &pool = m.register_pool();
    auto &mem = m.memory_move();
    auto &mat = m.matmul();

    // Live registers
    const Reg args = a0;
    const Reg ptra = a1;
    const Reg ptrb = a2;
    const Reg ptrc = a3;
    const Reg K = a4;
    const Reg k = a5;
    const Reg lda = a6;
    const Reg ldb = a7;
    const Reg ldc = t0;
    const FReg alpha = fa0;
    const FReg beta = fa1;

    const SEW sew = sew_for(dt_c);
    const LMUL lm = LMUL::m4;

    pool.int_register_file_excluding(
            {args, ptra, ptrb, ptrc, K, k, lda, ldb, ldc});
    pool.float_register_file();

    // A addressing: TransA -> strided load (stride=lda) advancing ptra by sewba per step
    //               !TransA -> unit load advancing ptra by lda per step
    const const_t a_outer = isTransA_ ? const_t(sewba) : const_t(lda);
    const const_t a_inner = isTransA_ ? const_t(lda) : const_t(0);

    // B addressing: TransB -> 1 pivot advancing by ldb per k-step (Case B)
    //               !TransB -> N pivots each advancing by sewbb per k-step (Case C)
    const const_t b_outer = isTransB_ ? const_t(sewbb) : const_t(ldb);
    const const_t b_inner = isTransB_ ? const_t(ldb) : const_t(sewbb);

    mat.configure(N_UR, dt_a, dt_c, lm, ptra, a_outer, a_inner, ptrb, b_outer,
            b_inner, ptrc);

    // Temporaries
    const x_block_t tmp = pool.new_int(2);
    const Reg avl = tmp[0];
    const Reg N = tmp[0];
    const Reg Ntmp = tmp[1];
    const Reg Ktmp = tmp[0];
    const Reg beta_bits = tmp[0];
    const Reg bias_ptr = tmp[0];
    const VReg vtmp = mat.scratch_vreg();

    // Dispatch plan for micro-kernel: switch case (n)
    const auto plan = dispatch_plan_t::dispatch(N_UR, N, Ntmp);

    // Code start

    pool.preserve();

    // Prepare pointers
    ld(ptra, args, offsetof(call_params_t, A));
    ld(ptrb, args, offsetof(call_params_t, B));
    ld(ptrc, args, offsetof(call_params_t, C));

    // Prepare strides
    ld(lda, args, offsetof(call_params_t, lda));
    ld(ldb, args, offsetof(call_params_t, ldb));
    ld(ldc, args, offsetof(call_params_t, ldc));
    slli(lda, lda, math::ilog2q(sewba));
    slli(ldb, ldb, math::ilog2q(sewbb));
    slli(ldc, ldc, math::ilog2q(sewbc));

    // Prepare loop limits
    ld(N, args, offsetof(call_params_t, n));
    ld(K, args, offsetof(call_params_t, K));

    // Setup VPU
    ld(avl, args, offsetof(call_params_t, m));
    vsetvli(x0, avl, sew, lm, VTA::ta, VMA::ma);

    // Dispatch
    cf.dispatch(plan, [&](int n_unroll) {
        mat.dense_loop(n_unroll, K_UR, k, K, Ktmp);

        const v_block_t c = mat.c_data();

        // Post-ops
        lw(beta_bits, args, offsetof(call_params_t, beta));
        flw(alpha, args, offsetof(call_params_t, alpha));
        flw(beta, args, offsetof(call_params_t, beta));
        cf.if_nez(beta_bits, [&](bool nonzero) {
            for (int n = 0; n < n_unroll; n++) {
                if (nonzero) {
                    // beta != 0: result = alpha*acc + beta*C [+ bias]
                    mem.vle(vtmp, ptrc, dt_c);
                    vfmul_vf(vtmp, vtmp, beta);
                    vfmul_vf(c[n], c[n], alpha);
                    vfadd_vv(vtmp, vtmp, c[n]);
                    if (has_bias_) {
                        ld(bias_ptr, args, offsetof(call_params_t, bias));
                        cf.if_nez(bias_ptr, [&] {
                            mem.vle(c[n], bias_ptr, dt_c);
                            vfadd_vv(vtmp, vtmp, c[n]);
                        });
                    }
                    mem.vse(vtmp, ptrc, dt_c);
                } else {
                    // beta == 0: result = alpha*acc [+ bias]
                    vfmul_vf(c[n], c[n], alpha);
                    if (has_bias_) {
                        ld(bias_ptr, args, offsetof(call_params_t, bias));
                        cf.if_nez(bias_ptr, [&] {
                            mem.vle(vtmp, bias_ptr, dt_c);
                            vfadd_vv(c[n], c[n], vtmp);
                        });
                    }
                    mem.vse(c[n], ptrc, dt_c);
                }
                add(ptrc, ptrc, ldc);
            }
        });
    });

    pool.restore();
    ret();
#else
    ret();
#endif
}

namespace {

template <bool isTransA, bool isTransB>
void jit_rvv_gemm_kernel_dispatch(const float *A, const float *B, float *C,
        dim_t lda, dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta,
        dim_t m, dim_t n, const float *bias) {
    static jit_rvv_gemm_kernel_t nb(isTransA, isTransB, false);
    static jit_rvv_gemm_kernel_t b(isTransA, isTransB, true);

    static bool verbose_printed = false;
    if (!verbose_printed) {
        VINFO(primitive, create, dispatch, rvv_gemm_jit,
                "JIT gemm kernel taking over: m=%d, n=%d", (int)m, (int)n);
        verbose_printed = true;
    }

    jit_rvv_gemm_kernel_t::call_params_t p;
    p.A = A;
    p.B = B;
    p.C = C;
    p.lda = lda;
    p.ldb = ldb;
    p.ldc = ldc;
    p.K = K;
    p.m = m;
    p.n = n;
    p.alpha = alpha;
    p.beta = beta;
    p.bias = bias;

    auto *kernel = bias ? &b : &nb;
    (*kernel)(&p);
}

} // namespace

void jit_rvv_gemm_kernel(const float *A, const float *B, float *C, dim_t lda,
        dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta, dim_t m,
        dim_t n, bool isTransA, bool isTransB, const float *bias) {
    if (!isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else if (isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<true, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else if (!isTransA && isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else {
        jit_rvv_gemm_kernel_dispatch<true, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
