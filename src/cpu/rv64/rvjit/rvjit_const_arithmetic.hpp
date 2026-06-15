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

#ifndef CPU_RV64_RVJIT_RVJIT_CONST_ARITHMETIC_HPP
#define CPU_RV64_RVJIT_RVJIT_CONST_ARITHMETIC_HPP

#include <iterator>
#include <initializer_list>

#include "common/math_utils.hpp"
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

/// Constant value during code generation stored either on reg or imm12 field
struct const_t {

    const_t() : kind(kind_t::imm), data(data_t()) {}
    const_t(const Reg &reg) : kind(kind_t::reg), data(reg) {}
    const_t(int imm) : kind(kind_t::imm), data(imm) {}

    bool is_reg() const { return kind == kind_t::reg; }
    bool is_imm() const { return kind == kind_t::imm; }
    Reg reg() const { return data.reg; }
    int imm() const { return data.imm; }

    bool is_valid() const { return is_reg() || (is_imm() && is_simm12(imm())); }

    bool is_zero() const {
        static const auto rzero = Xbyak_riscv::zero;
        return (is_imm() && !imm()) || (is_reg() && reg() == rzero);
    }

protected:
    enum class kind_t { reg, imm };

    union data_t {
        Reg reg;
        int imm;

        data_t() { imm = 0; }
        data_t(const Reg &r) { reg = r; }
        data_t(int i) { imm = i; }
    };

    kind_t kind;
    data_t data;
};

/// Component for constant-folded arithmetic
class const_arithmetic_t {
public:
    explicit const_arithmetic_t(emitter_t e) : e_(e) {}

    /// Prepares a code-generation constant value to be used
    ///
    /// @details Uses `tmp` as storage if c cannot be represented as a simm12
    const_t init_constant(int c, const Reg &tmp) const {
        if (is_simm12(c)) return const_t(c);
        e_->li(tmp, c);
        return const_t(tmp);
    }

    /// rd = rs1 + c
    ///
    /// @note The constant can reside on either a register or imm12 field
    /// @note Preserves the `rs1` register contents
    /// @note Nothing is emitted if the constant is detected to be zero
    void add_const(const Reg &rd, const Reg &rs1, const const_t &c) const {
        if (!c.is_zero()) {
            if (c.is_imm())
                e_->addi(rd, rs1, c.imm());
            else
                e_->add(rd, rs1, c.reg());
        }
    }

    /// rd = floor(rs1 / c)
    ///
    /// @pre `rd` must not alias `rs1` if the c value is not is_power2(abs(c))
    ///
    /// @note Preserves the `c.reg` register contents
    /// @note Preserves the `rs1` register contents
    void div_const(const Reg &rd, const Reg &rs1, const const_t &c) const {
        // Reject divison by zero
        if (c.is_zero()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed const_arithmetic_t.div_const (div by "
                        "zero)\n");
            });
            return;
        }

        // Constant from register
        if (c.is_reg()) {
            e_->div(rd, rs1, c.reg());
            return;
        }

        // Constant from immediate
        const auto imm = c.imm();
        const auto abs_imm = imm < 0 ? -imm : imm;

        if (math::is_pow2(abs_imm)) {
            // Quick division if power of two
            const int shammt = math::ilog2q(abs_imm);
            e_->srli(rd, rs1, shammt);
            if (imm < 0) e_->sub(rd, Xbyak_riscv::zero, rd);
        } else if (rd != rs1) {
            // Fallback to div if rd and rs1 are not aliased
            e_->li(rd, imm);
            e_->div(rd, rs1, rd);
            return;
        } else {
            // Illegal alias of rd and rs1
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed const_arithmetic_t.div_const due to "
                        "rd alias\n");
            });
        }
    }

    /// rd = rs1 - (rs1 % c)
    ///
    /// @pre `rd` must not alias `rs1` if c value is not is_power2(abs(c))
    ///
    /// @note This method preserves the `c.reg` register contents
    /// @note This method preserves the `rs1` register contents
    void round_down(const Reg &rd, const Reg &rs1, const const_t &c) const {
        // Reject divison by zero
        if (c.is_zero()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed const_arithmetic_t.round_down due to "
                        "division by zero\n");
            });
            return;
        }

        // Constant from register
        if (c.is_reg()) {
            e_->div(rd, rs1, c.reg());
            e_->mul(rd, rd, c.reg());
            return;
        }

        // Constant from immediate
        const auto imm = c.imm();
        const auto abs_imm = imm < 0 ? -imm : imm;

        if (math::is_pow2(abs_imm)) {
            // Quick division and multiplication
            const int shammt = math::ilog2q(abs_imm);
            e_->srli(rd, rs1, shammt);
            e_->slli(rd, rd, shammt);
        } else if (rd != rs1) {
            // Default to rem
            e_->li(rd, imm);
            e_->rem(rd, rs1, rd);
            e_->sub(rd, rs1, rd);
        } else {
            // Illegal alias of rd and rs1
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "Failed const_arithmetic_t.round_down due to "
                        "rd and rs1 alias\n");
            });
        }
    }

private:
    emitter_t e_;
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif