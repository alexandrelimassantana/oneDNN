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

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

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

bool evaluate(cond_t cond, int64_t lhs, int64_t rhs) {
    switch (cond) {
        case cond_t::eq: return lhs == rhs;
        case cond_t::ne: return lhs != rhs;
        case cond_t::lt: return lhs < rhs;
        case cond_t::le: return lhs <= rhs;
        case cond_t::gt: return lhs > rhs;
        case cond_t::ge: return lhs >= rhs;
    }
    return false;
}

// Independent reference re-implementation of unrolled_loop()/unrolled_loops()
struct unroll_result_t {
    int64_t total = 0;
    int64_t passes = 0;
};

unroll_result_t simulate_unrolled_loops(
        const std::vector<int> &factors, int64_t n) {
    unroll_result_t r;
    int64_t iter = 0;
    for (int u : factors) {
        const int64_t limit = (u == 1) ? n : (n - n % u);
        while (iter < limit) {
            iter += u;
            r.total += u;
            r.passes += 1;
        }
    }
    return r;
}

} // namespace

// if then: branch taken iff the comparison holds
struct if_then_probe_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(if_then_probe_kernel_t)

    if_then_probe_kernel_t(cond_t cond, distance_t dist)
        : jit_generator_t("rv64_test_if_then_probe"), cond_(cond), dist_(dist) {
        create_kernel();
    }

    void operator()(int64_t lhs, int64_t rhs, int64_t *out) const {
        jit_generator_t::operator()(lhs, rhs, out);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();

        const Reg lhs = a0;
        const Reg rhs = a1;
        const Reg out = a2;
        const Reg res = a3;

        mv(res, x0);
        cf.if_(
                lhs, cond_, rhs, [&] { li(res, 1); }, dist_);
        sd(res, out, 0);
        ret();
    }

private:
    cond_t cond_;
    distance_t dist_;
};

class if_then_test_t
    : public ::testing::TestWithParam<
              std::tuple<cond_t, distance_t, int64_t, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(if_then_test_t, if_then) {
    cond_t cond;
    distance_t dist;
    int64_t lhs, rhs;
    std::tie(cond, dist, lhs, rhs) = GetParam();

    if_then_probe_kernel_t kernel(cond, dist);
    int64_t out = -1;
    kernel(lhs, rhs, &out);

    const int64_t expect = evaluate(cond, lhs, rhs) ? 1 : 0;
    ASSERT_EQ(out, expect);
}

INSTANTIATE_TEST_SUITE_P(AllConditions, if_then_test_t,
        ::testing::Combine(::testing::Values(cond_t::eq, cond_t::ne, cond_t::lt,
                                   cond_t::le, cond_t::gt, cond_t::ge),
                ::testing::Values(distance_t::short_, distance_t::medium),
                ::testing::Values<int64_t>(-5, 0, 5),
                ::testing::Values<int64_t>(-5, 0, 5)));

// if then else: exactly one path runs
struct if_then_else_probe_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(if_then_else_probe_kernel_t)

    explicit if_then_else_probe_kernel_t(cond_t cond)
        : jit_generator_t("rv64_test_if_then_else_probe"), cond_(cond) {
        create_kernel();
    }

    void operator()(int64_t lhs, int64_t rhs, int64_t *out) const {
        jit_generator_t::operator()(lhs, rhs, out);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();

        const Reg lhs = a0;
        const Reg rhs = a1;
        const Reg out = a2;
        const Reg res = a3;

        cf.if_(lhs, cond_, rhs, [&](bool taken) { li(res, taken ? 1 : 2); });
        sd(res, out, 0);
        ret();
    }

private:
    cond_t cond_;
};

class if_then_else_test_t
    : public ::testing::TestWithParam<std::tuple<cond_t, int64_t, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(if_then_else_test_t, if_else) {
    cond_t cond;
    int64_t lhs, rhs;
    std::tie(cond, lhs, rhs) = GetParam();

    if_then_else_probe_kernel_t kernel(cond);
    int64_t out = -1;
    kernel(lhs, rhs, &out);

    const int64_t expect = evaluate(cond, lhs, rhs) ? 1 : 2;
    ASSERT_EQ(out, expect);
}

INSTANTIATE_TEST_SUITE_P(AllConditions, if_then_else_test_t,
        ::testing::Combine(::testing::Values(cond_t::eq, cond_t::ne, cond_t::lt,
                                   cond_t::le, cond_t::gt, cond_t::ge),
                ::testing::Values<int64_t>(-5, 0, 5),
                ::testing::Values<int64_t>(-5, 0, 5)));

// while less than step: counts how many times the body runs
struct while_step_probe_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(while_step_probe_kernel_t)

    while_step_probe_kernel_t(cond_t cond, int step)
        : jit_generator_t("rv64_test_while_step_probe")
        , cond_(cond)
        , step_(step) {
        create_kernel();
    }

    void operator()(int64_t n, int64_t *count_out, int64_t *iter_out) const {
        jit_generator_t::operator()(n, count_out, iter_out);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();

        const Reg N = a0;
        const Reg out_count = a1;
        const Reg out_iters = a2;
        const Reg iters = a3;
        const Reg n = a4;

        mv(iters, x0);
        mv(n, x0);
        cf.while_(iters, cond_, N, const_t(step_), [&] { addi(n, n, 1); });
        sd(n, out_count, 0);
        sd(iters, out_iters, 0);
        ret();
    }

private:
    cond_t cond_;
    int step_;
};

class while_step_test_t
    : public ::testing::TestWithParam<std::tuple<int64_t, int>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(while_step_test_t, CountsAndFinalIterMatch) {
    int64_t n;
    int step;
    std::tie(n, step) = GetParam();

    while_step_probe_kernel_t kernel(cond_t::lt, step);
    int64_t count = -1;
    int64_t iters = -1;
    kernel(n, &count, &iters);

    const int64_t expect_count = (n + step - 1) / step;
    ASSERT_EQ(count, expect_count);
    ASSERT_EQ(iters, expect_count * step);
}

INSTANTIATE_TEST_SUITE_P(Sizes, while_step_test_t,
        ::testing::Combine(::testing::Values<int64_t>(0, 1, 5, 7, 10, 16),
                ::testing::Values(1, 2, 4)));

// unrolled loops: same total work, fewer passes on larger unrolling
struct unrolled_loops_probe_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(unrolled_loops_probe_kernel_t)

    explicit unrolled_loops_probe_kernel_t(std::vector<int> factors)
        : jit_generator_t("rv64_test_unrolled_loops_probe")
        , factors_(std::move(factors)) {
        create_kernel();
    }

    void operator()(int64_t n, int64_t *total_out, int64_t *passes_out) const {
        jit_generator_t::operator()(n, total_out, passes_out);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();

        const Reg n = a0;
        const Reg total_ptr = a1;
        const Reg passes_ptr = a2;
        const Reg iters = a3;
        const Reg total = a4;
        const Reg passes = a5;
        const Reg tmp = a6;

        mv(iters, x0);
        mv(total, x0);
        mv(passes, x0);
        cf.unrolled_loops(iters, n, tmp, factors_.begin(), factors_.end(),
                [&](int unroll) {
            addi(passes, passes, 1);
            addi(total, total, unroll);
        });
        sd(total, total_ptr, 0);
        sd(passes, passes_ptr, 0);
        ret();
    }

private:
    std::vector<int> factors_;
};

class unrolled_loops_test_t
    : public ::testing::TestWithParam<std::tuple<std::vector<int>, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(
        unrolled_loops_test_t, ConsistentTotalFewerPasses) {
    std::vector<int> factors;
    int64_t n;
    std::tie(factors, n) = GetParam();

    unrolled_loops_probe_kernel_t unrolled(factors);
    unrolled_loops_probe_kernel_t reference({1});

    int64_t unrolled_total = -1;
    int64_t unrolled_passes = -1;
    unrolled(n, &unrolled_total, &unrolled_passes);

    int64_t reference_total = -1;
    int64_t reference_passes = -1;
    reference(n, &reference_total, &reference_passes);

    const auto expect = simulate_unrolled_loops(factors, n);

    // Unrolling never changes the amount of work done
    ASSERT_EQ(unrolled_total, n);
    ASSERT_EQ(reference_total, n);
    ASSERT_EQ(unrolled_total, reference_total);

    // Matches an independent reference implementation
    ASSERT_EQ(unrolled_passes, expect.passes);

    // Fewer passes once some non-trivial factor can engage. With more than
    // one real factor (e.g. {8, 4, 1}), a smaller later factor can still
    // produce savings even if the largest one in the list never gets the
    // chance to run, so compare against the smallest non-1 factor.
    int min_factor = *std::max_element(factors.begin(), factors.end());
    for (int f : factors)
        if (f > 1) min_factor = std::min(min_factor, f);
    if (n >= min_factor) {
        ASSERT_LT(unrolled_passes, reference_passes);
    } else {
        ASSERT_EQ(unrolled_passes, reference_passes);
    }
}

INSTANTIATE_TEST_SUITE_P(FactorLists, unrolled_loops_test_t,
        ::testing::Combine(
                ::testing::Values(std::vector<int> {4, 1},
                        std::vector<int> {7, 1}, std::vector<int> {8, 4, 1},
                        std::vector<int> {3, 1}),
                ::testing::Values<int64_t>(
                        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17)));

// switch_case: exactly the matching case executes
struct switch_case_probe_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(switch_case_probe_kernel_t)

    explicit switch_case_probe_kernel_t(int n_cases)
        : jit_generator_t("rv64_test_switch_case_probe"), n_cases_(n_cases) {
        create_kernel();
    }

    void operator()(int64_t id, int64_t *out) const {
        jit_generator_t::operator()(id, out);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();

        const Reg id = a0;
        const Reg out = a1;
        const Reg tmp = a2;
        const Reg result = a4;

        mv(result, x0);
        cf.switch_case(n_cases_, id, tmp, [&](int i) { li(result, i); });
        sd(result, out, 0);
        ret();
    }

private:
    int n_cases_;
};

class switch_case_test_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(switch_case_test_t, ExactlyMatchingCaseRuns) {
    int n_cases;
    int64_t id;
    std::tie(n_cases, id) = GetParam();

    switch_case_probe_kernel_t kernel(n_cases);
    int64_t out = -1;
    kernel(id, &out);

    ASSERT_EQ(out, id);
}

INSTANTIATE_TEST_SUITE_P(CaseCounts, switch_case_test_t,
        ::testing::ValuesIn(std::vector<std::tuple<int, int64_t>> {
                {1, 1},
                {3, 1},
                {3, 2},
                {3, 3},
                {16, 1},
                {16, 8},
                {16, 16},
        }));

// dispatch(): one probe per technique, all verify total coverage (total == N)

namespace {

// literal: cb called once with the compile-time value; total = n, passes = 1
struct dispatch_literal_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(dispatch_literal_probe_t)

    explicit dispatch_literal_probe_t(int n)
        : jit_generator_t("dispatch_literal_probe"), n_(n) {
        create_kernel();
    }

    void operator()(int64_t *total, int64_t *passes) const {
        jit_generator_t::operator()(total, passes);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();
        const Reg total_ptr = a0, passes_ptr = a1;
        const Reg total = a2, passes = a3;
        mv(total, x0);
        mv(passes, x0);
        cf.dispatch(dispatch_plan_t::literal(n_), [&](int nb) {
            addi(passes, passes, 1);
            addi(total, total, nb);
        });
        sd(total, total_ptr, 0);
        sd(passes, passes_ptr, 0);
        ret();
    }

private:
    int n_;
};

// dispatch: switch_case on a runtime id; total = id, passes = 1
struct dispatch_switch_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(dispatch_switch_probe_t)

    explicit dispatch_switch_probe_t(int max_n)
        : jit_generator_t("dispatch_switch_probe"), max_n_(max_n) {
        create_kernel();
    }

    void operator()(int64_t id, int64_t *total, int64_t *passes) const {
        jit_generator_t::operator()(id, total, passes);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();
        const Reg id = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4, scratch = a5;
        mv(total, x0);
        mv(passes, x0);
        cf.dispatch(dispatch_plan_t::dispatch(max_n_, id, scratch),
                [&](int nb) {
            addi(passes, passes, 1);
            addi(total, total, nb);
        });
        sd(total, total_ptr, 0);
        sd(passes, passes_ptr, 0);
        ret();
    }

private:
    int max_n_;
};

// greedy: unrolled_loops {preferred, 1}; total always == limit
struct dispatch_greedy_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(dispatch_greedy_probe_t)

    explicit dispatch_greedy_probe_t(int preferred)
        : jit_generator_t("dispatch_greedy_probe"), preferred_(preferred) {
        create_kernel();
    }

    void operator()(int64_t n, int64_t *total, int64_t *passes) const {
        jit_generator_t::operator()(n, total, passes);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();
        const Reg n = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4, iter = a5, tmp = a6;
        mv(total, x0);
        mv(passes, x0);
        cf.dispatch(dispatch_plan_t::greedy(preferred_, iter, n, tmp),
                [&](int nb) {
            addi(passes, passes, 1);
            addi(total, total, nb);
        });
        sd(total, total_ptr, 0);
        sd(passes, passes_ptr, 0);
        ret();
    }

private:
    int preferred_;
};

// main_then_dispatch: preferred-wide main + switch_case tail; total always == limit
struct dispatch_main_then_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(dispatch_main_then_probe_t)

    explicit dispatch_main_then_probe_t(int preferred)
        : jit_generator_t("dispatch_main_then_probe"), preferred_(preferred) {
        create_kernel();
    }

    void operator()(int64_t n, int64_t *total, int64_t *passes) const {
        jit_generator_t::operator()(n, total, passes);
    }

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &cf = m.control_flow();
        const Reg n = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4, iter = a5, tmp = a6;
        mv(total, x0);
        mv(passes, x0);
        cf.dispatch(
                dispatch_plan_t::main_then_dispatch(preferred_, iter, n, tmp),
                [&](int nb) {
            addi(passes, passes, 1);
            addi(total, total, nb);
                });
        sd(total, total_ptr, 0);
        sd(passes, passes_ptr, 0);
        ret();
    }

private:
    int preferred_;
};

int64_t expected_greedy_passes(int preferred, int64_t n) {
    return n / preferred + n % preferred;
}

int64_t expected_main_then_passes(int preferred, int64_t n) {
    return n / preferred + (n % preferred > 0 ? 1 : 0);
}

} // namespace

TEST(DispatchLiteral, TotalEqualsN) {
    for (int n : {1, 4, 7}) {
        dispatch_literal_probe_t k(n);
        int64_t total = -1, passes = -1;
        k(&total, &passes);
        EXPECT_EQ(total, n) << "n=" << n;
        EXPECT_EQ(passes, 1) << "n=" << n;
    }
}

class dispatch_switch_test_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(dispatch_switch_test_t, TotalEqualsId) {
    int max_n;
    int64_t id;
    std::tie(max_n, id) = GetParam();

    dispatch_switch_probe_t k(max_n);
    int64_t total = -1, passes = -1;
    k(id, &total, &passes);
    EXPECT_EQ(total, id);
    EXPECT_EQ(passes, 1);
}

INSTANTIATE_TEST_SUITE_P(Combos, dispatch_switch_test_t,
        ::testing::ValuesIn(std::vector<std::tuple<int, int64_t>> {
                {4, 1}, {4, 2}, {4, 3}, {4, 4}, {6, 1}, {6, 6}}));

class dispatch_greedy_test_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(dispatch_greedy_test_t, TotalEqualsN) {
    int preferred;
    int64_t n;
    std::tie(preferred, n) = GetParam();

    dispatch_greedy_probe_t k(preferred);
    int64_t total = -1, passes = -1;
    k(n, &total, &passes);
    EXPECT_EQ(total, n);
    EXPECT_EQ(passes, expected_greedy_passes(preferred, n));
}

INSTANTIATE_TEST_SUITE_P(Combos, dispatch_greedy_test_t,
        ::testing::Combine(::testing::Values(1, 4, 6),
                ::testing::Values<int64_t>(0, 1, 3, 4, 5, 7, 8, 12)));

class dispatch_main_then_test_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(dispatch_main_then_test_t, TotalEqualsN) {
    int preferred;
    int64_t n;
    std::tie(preferred, n) = GetParam();

    dispatch_main_then_probe_t k(preferred);
    int64_t total = -1, passes = -1;
    k(n, &total, &passes);
    EXPECT_EQ(total, n);
    EXPECT_EQ(passes, expected_main_then_passes(preferred, n));
}

INSTANTIATE_TEST_SUITE_P(Combos, dispatch_main_then_test_t,
        ::testing::Combine(::testing::Values(1, 4, 6),
                ::testing::Values<int64_t>(0, 1, 3, 4, 5, 7, 8, 12)));

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
