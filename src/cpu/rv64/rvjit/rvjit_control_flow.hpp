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

#ifndef CPU_RV64_RVJIT_RVJIT_CONTROL_FLOW_HPP
#define CPU_RV64_RVJIT_RVJIT_CONTROL_FLOW_HPP

#include <initializer_list>

#include "cpu/rv64/rvjit/rvjit_const_arithmetic.hpp"
#include "cpu/rv64/rvjit/rvjit_emitter.hpp"

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

// Callback type to emit code for a conditional basic block
using on_cond_t = callback_t<bool>;

// Callback type to emit code for a code block unrolled by some amount
using on_unrolled_t = callback_t<int>;

// Callback type to emit code for an indexed basic block
//
/// @details The first argument is the block index
using on_case_t = callback_t<int>;

/// Branch condition types
enum class cond_t { eq, ne, lt, le, gt, ge };

/// Dispatch technique for N-tile loop dispatch
enum class dispatch_technique_t {
    literal, ///< compile-time constant: cb(n) once, no branch
    dispatch, ///< runtime switch_case on a pre-loaded id register
    greedy, ///< unrolled_loops {preferred, 1}: main tiles + 1-wide tail
    main_then_dispatch ///< main loop of preferred + switch_case for the tail
};

/// Branch distance type short uses a direct branch instruction,
/// medium emits a jump-and-link to reach targets beyond a distance of 4 KB.
enum class distance_t { short_, medium };

namespace condition {
constexpr cond_t eq = cond_t::eq; // ==
constexpr cond_t ne = cond_t::ne; // !=
constexpr cond_t lt = cond_t::lt; // <
constexpr cond_t le = cond_t::le; // <=
constexpr cond_t gt = cond_t::gt; // >
constexpr cond_t ge = cond_t::ge; // >=
} // namespace condition

namespace distance {
constexpr distance_t short_ = distance_t::short_; // Fits a 12-bit simm
constexpr distance_t medium = distance_t::medium; // Fits a 20-bit simm
} // namespace distance

/// Describes how to dispatch N-tile micro-kernel invocations
///
/// @note Similar to vaddr_t: choose a factory method, pass to dispatch()
struct dispatch_plan_t {
    dispatch_technique_t technique = dispatch_technique_t::literal;

    int n = 0; ///< literal: value; dispatch: max_n; greedy/m_t_d: preferred
    Reg id {}, scratch {}; ///< dispatch: runtime id / scratch registers
    Reg iter {}, limit {},
            tmp {}; ///< greedy/m_t_d: loop counter, bound, scratch
    distance_t d = distance_t::
            short_; ///< branch distance for emitted loops/branches

    /// cb(n) once, no runtime branching
    static dispatch_plan_t literal(int n) {
        dispatch_plan_t p;
        p.technique = dispatch_technique_t::literal;
        p.n = n;
        return p;
    }

    /// switch_case(max_n, id, scratch, cb) — id must already be in [1, max_n]
    static dispatch_plan_t dispatch(int max_n, Reg id, Reg scratch) {
        dispatch_plan_t p;
        p.technique = dispatch_technique_t::dispatch;
        p.n = max_n;
        p.id = id;
        p.scratch = scratch;
        return p;
    }

    /// unrolled_loops {preferred, 1}: floor(limit/preferred) main + tail 1-wide steps
    static dispatch_plan_t greedy(int preferred, Reg iter, Reg limit, Reg tmp,
            distance_t d = distance::short_) {
        dispatch_plan_t p;
        p.technique = dispatch_technique_t::greedy;
        p.n = preferred;
        p.iter = iter;
        p.limit = limit;
        p.tmp = tmp;
        p.d = d;
        return p;
    }

    /// floor(limit/preferred) main steps + switch_case for tail (if any)
    ///
    /// @note iter is reused as switch_case scratch after the main loop completes
    /// @note preferred must be ≤ 17 so that the tail switch_case N = preferred-1 ≤ 16
    static dispatch_plan_t main_then_dispatch(int preferred, Reg iter,
            Reg limit, Reg tmp, distance_t d = distance::short_) {
        dispatch_plan_t p;
        p.technique = dispatch_technique_t::main_then_dispatch;
        p.n = preferred;
        p.iter = iter;
        p.limit = limit;
        p.tmp = tmp;
        p.d = d;
        return p;
    }

    static bool dispatch_plan_is_valid(const dispatch_plan_t &p) {
        using dt = dispatch_technique_t;
        switch (p.technique) {
            case dt::literal: return p.n > 0;
            case dt::dispatch:
                return p.n >= 1 && p.n <= 16 && p.id != Xbyak_riscv::x0
                        && p.id != p.scratch;
            case dt::greedy:
                return p.n >= 1 && p.iter != Xbyak_riscv::x0
                        && p.limit != Xbyak_riscv::x0
                        && p.tmp != Xbyak_riscv::x0 && p.iter != p.limit
                        && p.tmp != p.limit;
            case dt::main_then_dispatch:
                return p.n >= 2 && p.iter != Xbyak_riscv::x0
                        && p.limit != Xbyak_riscv::x0
                        && p.tmp != Xbyak_riscv::x0 && p.iter != p.limit
                        && p.tmp != p.limit && p.iter != p.tmp;
        }
        return false;
    }
};

/// Component expressing common control flow patterns.
class control_flow_t {
public:
    control_flow_t(emitter_t e, const const_arithmetic_t &arith)
        : e_(e), arith_(arith) {}

    /// If-then pattern emitter
    ///
    /// @details Preserves the `lhs` and `rhs` registers
    void if_(const Reg &lhs, cond_t c, const Reg &rhs, const on_enter_t &body,
            distance_t d = distance::short_) {
        Label skip;
        emit_branch_(lhs, rhs, c, d, skip);
        body();
        e_->L(skip);
    }

    inline void if_eqz(const Reg &lhs, const on_enter_t &cb) {
        if_(lhs, condition::eq, Xbyak_riscv::zero, cb);
    }
    inline void if_nez(const Reg &lhs, const on_enter_t &cb) {
        if_(lhs, condition::ne, Xbyak_riscv::zero, cb);
    }

    /// If-then-else pattern emitter
    ///
    /// @details Preserves the `lhs` and `rhs` registers
    void if_(const Reg &lhs, cond_t c, const Reg &rhs, const on_cond_t &body,
            distance_t d = distance::short_) {
        if (!body) return;
        Label skip, done;
        emit_branch_(lhs, rhs, c, d, skip);
        body(true);
        e_->j_(done);
        e_->L(skip);
        body(false);
        e_->L(done);
    }

    inline void if_eqz(const Reg &lhs, const on_cond_t &cb) {
        if_(lhs, condition::eq, Xbyak_riscv::zero, cb);
    }
    inline void if_nez(const Reg &lhs, const on_cond_t &cb) {
        if_(lhs, condition::ne, Xbyak_riscv::zero, cb);
    }

    /// While loop pattern emitter
    ///
    /// @pre Caller must ensure `iter` progress in `body`
    ///
    /// @details Preserves the `iter` and `limit` registers
    void while_(const Reg &iter, cond_t c, const Reg &limit,
            const on_enter_t &body, distance_t d = distance::short_) {
        if (!body) return;
        Label head, end;
        e_->L(head);
        emit_branch_(iter, limit, c, d, end);
        body();
        e_->j_(head);
        e_->L(end);
    }

    inline void while_lt(const Reg &lhs, const Reg &rhs, const on_enter_t &body,
            distance_t d = distance::short_) {
        while_(lhs, condition::lt, rhs, body, d);
    }

    /// While loop with step pattern emitter
    ///
    /// @note Emits code to advance `iter` using the `step` constant
    ///
    /// @pre Caller must preserve `iter` if clobbered in `body`
    ///
    /// @details The `iter` value is updated before calling `body`
    /// @details Preserves the `end` register
    /// @details Clobbers the `iter` register
    void while_(const Reg &iter, cond_t c, const Reg &limit, const const_t step,
            const on_enter_t &body, distance_t d = distance::short_) {
        if (!body) return;
        Label head, end;
        e_->L(head);
        emit_branch_(iter, limit, c, d, end);
        arith_.add_const(iter, iter, step);
        body();
        e_->j_(head);
        e_->L(end);
    }

    inline void while_lt(const Reg &iter, const Reg &end, const const_t step,
            const on_enter_t &body, distance_t d = distance::short_) {
        while_(iter, condition::lt, end, step, body, d);
    }

    /// Emit a while_lt loop unrolled by a certain number of iterations
    ///
    /// @pre Caller must preserve `tmp` and `to` if clobbered in `body`
    ///
    /// @details Preserves the `to` register
    void unrolled_loop(const Reg &from, const Reg &to, const Reg &tmp,
            int unroll, on_unrolled_t body, distance_t d = distance::short_) {
        if (unroll == 1) {
            while_lt(
                    from, to, unroll, [&] { body(unroll); }, d);
        } else if (unroll > 1) {
            arith_.round_down(tmp, to, unroll);
            while_lt(
                    from, tmp, unroll, [&] { body(unroll); }, d);
        }
    }

    /// Emit a series of while_lt loops, each with its own unroll factor
    ///
    /// @pre Caller must ensure iterators are sorted in descending order
    /// @pre Caller must preserve `tmp` and `to` if clobbered in `body`
    ///
    /// @details Preserves the `to` register
    template <typename It>
    void unrolled_loops(const Reg &from, const Reg &to, const Reg &tmp, It us,
            It ue, on_unrolled_t body, distance_t d = distance::short_) {
        for (; us != ue; ++us)
            unrolled_loop(from, to, tmp, *us, body, d);
    }

    /// Emit a series of while_lt loops, each individually unrolled
    ///
    /// @pre Caller must ensure iterators are sorted in descending order
    /// @pre Caller must preserve `tmp` and `to` if clobbered in `body`
    ///
    /// @details Preserves the `to` register
    void unrolled_loops(const Reg &from, const Reg &to, const Reg &tmp,
            std::initializer_list<int> ur, on_unrolled_t body,
            distance_t d = distance::short_) {
        unrolled_loops(from, to, tmp, ur.begin(), ur.end(), body, d);
    }

    /// Switch-case emitter
    ///
    /// @pre Caller must ensure `id` fits the range [1, N] (N <= 16)
    ///
    /// @note Covers runtime `id` values between [1, N] (N <= 16).
    ///
    /// @details Clobbers the `id` and `t` registers
    void switch_case(int N, const Reg &id, const Reg &t, const on_case_t &cb) {
        static constexpr int MAXN = 16;
        if (N <= 0 || N > MAXN || id == Xbyak_riscv::x0) return;

        Label end;
        Label cases[MAXN];
        const Reg &address = t;
        const Reg &entry = id;

        // Byte offset from auipc to the selected case entry.
        // The jump table starts 3 instructions after auipc; entry i maps to
        // case N-i. offset = ((3 + N) * 4) - (id * 4)
        e_->li(address, (3 + N) * static_cast<int>(sizeof(uint32_t)));
        e_->slli(entry, id, 2);
        e_->sub(entry, address, entry);

        e_->auipc(address, 0);
        e_->add(address, address, entry);
        e_->jr(address);

        for (int i = N; i > 0; --i)
            e_->j_(cases[i - 1]);

        for (int i = N; i > 0; --i) {
            e_->L(cases[i - 1]);
            cb(i);
            if (i > 1) e_->j_(end);
        }

        e_->L(end);
    }

    /// Emits micro-kernel dispatch according to a dispatch_plan_t
    void dispatch(const dispatch_plan_t &p, const on_case_t &cb) {
        if (!dispatch_plan_t::dispatch_plan_is_valid(p)) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "control_flow_t::dispatch: invalid plan, emitting "
                        "nothing\n");
            });
            return;
        }
        using dt = dispatch_technique_t;
        switch (p.technique) {
            case dt::literal: cb(p.n); break;
            case dt::dispatch: switch_case(p.n, p.id, p.scratch, cb); break;
            case dt::greedy:
                e_->mv(p.iter, Xbyak_riscv::zero);
                unrolled_loops(
                        p.iter, p.limit, p.tmp, {p.n, 1},
                        [&](int nb) { cb(nb); }, p.d);
                break;
            case dt::main_then_dispatch:
                e_->mv(p.iter, Xbyak_riscv::zero);
                unrolled_loop(
                        p.iter, p.limit, p.tmp, p.n, [&](int nb) { cb(nb); },
                        p.d);
                e_->sub(p.tmp, p.limit, p.iter);
                if_(
                        p.tmp, condition::ne, Xbyak_riscv::zero,
                        [&] { switch_case(p.n - 1, p.tmp, p.iter, cb); }, p.d);
                break;
        }
    }

private:
    emitter_t e_;
    const const_arithmetic_t &arith_;

    void emit_branch_(const Reg &lhs, const Reg &rhs, cond_t c, distance_t d,
            const Label &not_taken) {

        using branch_insn_t
                = void (codegen_t::*)(const Reg &, const Reg &, const Label &);

        // Organized in the same order elements are declared
        static constexpr branch_insn_t branch_table[] = {
                &codegen_t::beq,
                &codegen_t::bne,
                &codegen_t::blt,
                &codegen_t::ble,
                &codegen_t::bgt,
                &codegen_t::bge,
        };

        // Emit a condition-agnostic branch instruction
        codegen_t &cg = e_.cg();
        const auto emit = [&](const cond_t &cond, const Label &lbl) {
            (cg.*branch_table[static_cast<int>(cond)])(lhs, rhs, lbl);
        };

        // Get the reverse branching condition
        const auto inverse = [&]() {
            switch (c) {
                case cond_t::eq: return cond_t::ne;
                case cond_t::ne: return cond_t::eq;
                case cond_t::lt: return cond_t::ge;
                case cond_t::ge: return cond_t::lt;
                case cond_t::le: return cond_t::gt;
                case cond_t::gt: return cond_t::le;
                default: return cond_t::ne;
            }
        };

        if (d == distance_t::short_) {
            // emits the inverse condition as a single branch instruction
            emit(inverse(), not_taken);
        } else {
            // Use jump when "taken" path is further than the 4 KB branch range
            Label taken;
            emit(c, taken);
            e_->j_(not_taken);
            e_->L(taken);
        }
    }
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
