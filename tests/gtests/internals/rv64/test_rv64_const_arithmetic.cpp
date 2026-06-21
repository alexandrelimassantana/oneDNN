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
#include <tuple>

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

namespace {

enum class arith_op_t { add_const, div_const, round_down };

// `div_const`/`round_down` require `rd != rs1` whenever the constant is an
// immediate that is not a power of two (see rvjit_const_arithmetic.hpp); in
// every other case -- add_const, register-sourced constants, or
// power-of-two immediates -- aliasing rd with rs1 is safe.
bool alias_is_legal(arith_op_t op, int c) {
    if (op == arith_op_t::add_const) return true;
    if (!is_simm12(c)) return true;
    if (op == arith_op_t::round_down && c == 1) return true;
    const int64_t abs_c = c < 0 ? -static_cast<int64_t>(c) : c;
    return math::is_pow2(abs_c);
}

} // namespace

struct const_init_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(const_init_kernel_t)

    const_init_kernel_t(int c)
        : jit_generator_t("rv64_test_const_init"), c_(c) {
        create_kernel();
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &arith = m.const_arithmetic();

        arith.init_constant(c_, Xbyak_riscv::a0);
        ret();
    }

private:
    int c_;
};

class const_init_test_t : public ::testing::TestWithParam<std::tuple<int64_t>> {
};

// Test if initialization code is emitted only when value is not a simm12
HANDLE_EXCEPTIONS_FOR_TEST_P(const_init_test_t, ConstInit) {
    int c;
    std::tie(c) = GetParam();

    const_init_kernel_t kernel(c);

    // Check if more than 1 instruction (ret) was emitted
    const bool should_emit = is_simm12(c) == false;
    const bool emited_init = kernel.getSize() > sizeof(int32_t);

    ASSERT_EQ(should_emit, emited_init);
}

INSTANTIATE_TEST_SUITE_P(ConstInit, const_init_test_t,
        ::testing::Values(0, 16, -16, 2047, -2048, 2048, -2049));

struct const_arithmetic_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(const_arithmetic_kernel_t)

    const_arithmetic_kernel_t(int c, arith_op_t op, bool alias_rd = false)
        : jit_generator_t("rv64_test_arithmetic_const")
        , c_(c)
        , op_(op)
        , alias_rd_(alias_rd) {
        create_kernel();
    }

    void operator()(int64_t *rd, const int64_t rs1) const {
        jit_generator_t::operator()(rd, rs1);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &arith = m.const_arithmetic();

        const Reg ptr = Xbyak_riscv::a0;
        const Reg rs1 = Xbyak_riscv::a1;
        const Reg tmp = Xbyak_riscv::a2;
        const Reg rd = alias_rd_ ? rs1 : Xbyak_riscv::a3;

        const auto c = arith.init_constant(c_, tmp);

        switch (op_) {
            case arith_op_t::add_const: {
                arith.add_const(rd, rs1, c);
                break;
            }
            case arith_op_t::div_const: {
                arith.div_const(rd, rs1, c);
                break;
            }
            case arith_op_t::round_down: {
                arith.round_down(rd, rs1, c);
                break;
            }
        }
        sd(rd, ptr);
        ret();
    }

private:
    int c_;
    arith_op_t op_;
    bool alias_rd_;
};

class const_arithmetic_test_t
    : public ::testing::TestWithParam<std::tuple<int64_t, arith_op_t>> {};

// Check arithmetic correctness when using the API properly (rd != rs1)
HANDLE_EXCEPTIONS_FOR_TEST_P(const_arithmetic_test_t, MatchesReference) {
    static const int64_t base = 4096;

    int c;
    arith_op_t op;
    std::tie(c, op) = GetParam();
    const_arithmetic_kernel_t kernel(c, op);

    int64_t expected;
    switch (op) {
        case arith_op_t::add_const: {
            expected = base + c;
            break;
        }
        case arith_op_t::div_const: {
            expected = base / c;
            break;
        }
        case arith_op_t::round_down: {
            expected = base - (base % c);
            break;
        }
    }

    int64_t received;
    kernel(&received, base);

    ASSERT_EQ(expected, received);
}

INSTANTIATE_TEST_SUITE_P(AllOps, const_arithmetic_test_t,
        ::testing::Combine(
                ::testing::Values(1, -1, 16, -16, 2047, -2048, 2048, -2049),
                ::testing::Values(arith_op_t::add_const, arith_op_t::div_const,
                        arith_op_t::round_down)));

class const_arithmetic_alias_test_t
    : public ::testing::TestWithParam<std::tuple<int64_t, arith_op_t>> {};

// Check that aliasing rd with rs1 (div_const/round_down) either computes the
// correct result (pow2/register-sourced constants) or is a no-op.
HANDLE_EXCEPTIONS_FOR_TEST_P(
        const_arithmetic_alias_test_t, AliasedRdEqualsRs1) {
    static const int64_t base = 4096;

    int c;
    arith_op_t op;
    std::tie(c, op) = GetParam();
    const_arithmetic_kernel_t kernel(c, op, /*alias_rd=*/true);

    int64_t expected;
    if (!alias_is_legal(op, c)) {
        expected = base;
    } else {
        switch (op) {
            case arith_op_t::add_const: {
                expected = base + c;
                break;
            }
            case arith_op_t::div_const: {
                expected = base / c;
                break;
            }
            case arith_op_t::round_down: {
                expected = base - (base % c);
                break;
            }
        }
    }

    int64_t received;
    kernel(&received, base);

    ASSERT_EQ(expected, received);
}

INSTANTIATE_TEST_SUITE_P(DivAndRoundDown, const_arithmetic_alias_test_t,
        ::testing::Combine(::testing::Values(1, -1, 3, 5, 7, 16, -16, 2047,
                                   -2048, 2048, -2049),
                ::testing::Values(
                        arith_op_t::div_const, arith_op_t::round_down)));

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
