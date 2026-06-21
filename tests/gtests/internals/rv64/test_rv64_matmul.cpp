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

#include <cstdint>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

#if XBYAK_RISCV_V

namespace {

// Case A: b_outer and b_inner both immediates — single shared B pointer.
// BRGEMM one_ptrb path. Expects code to be emitted (non-trivial kernel).
struct matmul_case_a_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(matmul_case_a_probe_t)
    matmul_case_a_probe_t() : jit_generator_t("rv64_matmul_case_a_probe") {}

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        auto &mat = m.matmul();

        const Reg ptra = a1, ptrb = a2, ptrc = a3;
        const Reg k = t0, K = t1, Ktmp = t2;
        pool.int_register_file_excluding({a0, ptra, ptrb, ptrc, k, K, Ktmp});
        pool.float_register_file();

        mat.configure(2, data_type::f32, data_type::f32, LMUL::m1, ptra,
                const_t(4), const_t(0), // a_outer=4B imm, unit load
                ptrb, const_t(256), const_t(4), // b_outer=256B imm, b_inner=4B
                ptrc);

        pool.preserve();
        mat.dense_loop(2, 2, k, K, Ktmp);
        pool.restore();
        ret();
    }
};

// Case C: b_outer is a runtime register — one B pointer per N column.
// GEMM !TransB / BRGEMM !one_ptrb path.
struct matmul_case_c_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(matmul_case_c_probe_t)
    matmul_case_c_probe_t() : jit_generator_t("rv64_matmul_case_c_probe") {}

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        auto &mat = m.matmul();

        const Reg ptra = a1, ptrb = a2, ptrc = a3, ldb = a4;
        const Reg k = t0, K = t1, Ktmp = t2;
        pool.int_register_file_excluding(
                {a0, ptra, ptrb, ptrc, ldb, k, K, Ktmp});
        pool.float_register_file();

        // b_outer=ldb (reg) → N pivots; b_inner=4B (imm) → batch advance
        mat.configure(2, data_type::f32, data_type::f32, LMUL::m1, ptra,
                const_t(4), const_t(0), ptrb, const_t(ldb), const_t(4), ptrc);

        pool.preserve();
        mat.dense_loop(2, 2, k, K, Ktmp);
        pool.restore();
        ret();
    }
};

// Widening f16->f32 with Case A (one_ptrb-style).
struct matmul_widening_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(matmul_widening_probe_t)
    matmul_widening_probe_t() : jit_generator_t("rv64_matmul_widening_probe") {}

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        auto &mat = m.matmul();

        const Reg ptra = a1, ptrb = a2, ptrc = a3;
        const Reg k = t0, K = t1, Ktmp = t2;
        pool.int_register_file_excluding({a0, ptra, ptrb, ptrc, k, K, Ktmp});
        pool.float_register_file();

        mat.configure(2, data_type::f16, data_type::f32, LMUL::m1, ptra,
                const_t(2), const_t(0), // a_outer=2B (f16 element)
                ptrb, const_t(128), const_t(2), // b_outer=128B imm, b_inner=2B
                ptrc);

        pool.preserve();
        mat.dense_loop(2, 2, k, K, Ktmp);
        pool.restore();
        ret();
    }
};

} // namespace

HANDLE_EXCEPTIONS_FOR_TEST(MatmulCaseA, EmitsNonTrivialKernel) {
    matmul_case_a_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    ASSERT_GT(k.getSize(), sizeof(uint32_t));
}

HANDLE_EXCEPTIONS_FOR_TEST(MatmulCaseC, EmitsNonTrivialKernel) {
    matmul_case_c_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    ASSERT_GT(k.getSize(), sizeof(uint32_t));
}

HANDLE_EXCEPTIONS_FOR_TEST(MatmulWidening, EmitsNonTrivialKernel) {
    matmul_widening_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    ASSERT_GT(k.getSize(), sizeof(uint32_t));
}

#endif // XBYAK_RISCV_V

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
