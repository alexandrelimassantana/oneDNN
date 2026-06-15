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

#ifndef CPU_RV64_RVJIT_RVJIT_FMA_HPP
#define CPU_RV64_RVJIT_RVJIT_FMA_HPP

#include "common/c_types_map.hpp"

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

/// Describes a fused multiply-accumulate's operand dtypes
struct fma_t {
    data_type_t acc, s1, s2;

    fma_t(data_type_t acc, data_type_t s1, data_type_t s2)
        : acc(acc), s1(s1), s2(s2) {}

    /// Same-precision: acc, s1, s2 all share `dt`
    static fma_t uniform(data_type_t dt) { return fma_t(dt, dt, dt); }

    /// Widening: s1 and s2 share the narrow `dt`; acc is a wider counterpart
    static fma_t widening(data_type_t dt) {
        return fma_t(natural_wide(dt), dt, dt);
    }

    bool is_uniform() const {
        return sewb(acc) == sewb(s1) && sewb(s1) == sewb(s2);
    }

    bool is_widening() const {
        return sewb(s1) == sewb(s2) && sewb(acc) == 2 * sewb(s1);
    }
};

/// Base for type-adaptive multiply-accumulate emitters
class macc_emitter_t {
public:
    explicit macc_emitter_t(emitter_t e) : e_(e) {}

protected:
    emitter_t e_;

    static void log_unsupported(const char *caller) {
        DEBUG({
            verbose_printf(verbose_t::debuginfo,
                    "Failed %s due to unsupported fma_t\n", caller);
        });
    }
};

/// Multiply-accumulate emitter for RVV and extended platforms
class rvv_macc_emitter_t : public macc_emitter_t {
public:
    explicit rvv_macc_emitter_t(emitter_t e) : macc_emitter_t(e) {}

    /// Integer multiplty-accumulate with vector operands
    void fmacc_int(const VReg &vd, const VReg &vs1, const VReg &vs2,
            const fma_t &op, VM vm = VM::unmasked) const {
        if (!is_int(op.s1) || !is_int(op.s2)) {
            log_unsupported("rvv_macc_emitter_t.fmacc_int");
            return;
        }
        if (op.is_uniform()) {
            e_->vmacc_vv(vd, vs1, vs2, vm);
            return;
        }
        if (op.is_widening()) {
            const bool s1_signed = is_signed_int(op.s1);
            const bool s2_signed = is_signed_int(op.s2);
            if (s1_signed && s2_signed) {
                e_->vwmacc_vv(vd, vs1, vs2, vm);
                return;
            }
            if (!s1_signed && !s2_signed) {
                e_->vwmaccu_vv(vd, vs1, vs2, vm);
                return;
            }
            if (s1_signed && !s2_signed) {
                e_->vwmaccsu_vv(vd, vs1, vs2, vm);
                return;
            }
        }
        log_unsupported("rvv_macc_emitter_t.fmacc_int");
    }

    /// Integer multiplty-accumulate with implicit broadcast
    void fmacc_int(const VReg &vd, const Reg &rs1, const VReg &vs2,
            const fma_t &op, VM vm = VM::unmasked) const {
        if (!is_int(op.s1) || !is_int(op.s2)) {
            log_unsupported("rvv_macc_emitter_t.macc_vx");
            return;
        }
        if (op.is_uniform()) {
            e_->vmacc_vx(vd, rs1, vs2, vm);
            return;
        }
        if (op.is_widening()) {
            const bool s1_signed = is_signed_int(op.s1);
            const bool s2_signed = is_signed_int(op.s2);
            if (s1_signed && s2_signed) {
                e_->vwmacc_vx(vd, rs1, vs2, vm);
                return;
            }
            if (!s1_signed && !s2_signed) {
                e_->vwmaccu_vx(vd, rs1, vs2, vm);
                return;
            }
            if (s1_signed && !s2_signed) {
                e_->vwmaccsu_vx(vd, rs1, vs2, vm);
                return;
            }
            if (!s1_signed && s2_signed) {
                e_->vwmaccus_vx(vd, rs1, vs2, vm);
                return;
            }
        }
        log_unsupported("rvv_macc_emitter_t.macc_vx");
    }

    /// Floating-point multiplty-accumulate with vector operands
    void fmacc_float(const VReg &vd, const VReg &vs1, const VReg &vs2,
            const fma_t &op, VM vm = VM::unmasked) const {
        using namespace data_type;
        if (op.s1 == op.s2) {
            if (op.is_uniform()
                    && (op.s1 == f16 || op.s1 == f32 || op.s1 == f64)) {
                e_->vfmacc_vv(vd, vs1, vs2, vm);
                return;
            }
            if (op.is_widening() && op.s1 == bf16) {
                e_->vfwmaccbf16_vv(vd, vs1, vs2, vm);
                return;
            }
            if (op.is_widening() && (op.s1 == f16 || op.s1 == f32)) {
                e_->vfwmacc_vv(vd, vs1, vs2, vm);
                return;
            }
        }
        log_unsupported("rvv_macc_emitter_t.fmacc_float");
    }

    /// Floating-point multiplty-accumulate with implicit broadcast
    void fmacc_float(const VReg &vd, const FReg &fs1, const VReg &vs2,
            const fma_t &op, VM vm = VM::unmasked) const {
        using namespace data_type;
        if (op.s1 == op.s2) {
            if (op.is_uniform()
                    && (op.s1 == f16 || op.s1 == f32 || op.s1 == f64)) {
                e_->vfmacc_vf(vd, fs1, vs2, vm);
                return;
            }
            if (op.is_widening() && op.s1 == bf16) {
                e_->vfwmaccbf16_vf(vd, fs1, vs2, vm);
                return;
            }
            if (op.is_widening() && (op.s1 == f16 || op.s1 == f32)) {
                e_->vfwmacc_vf(vd, fs1, vs2, vm);
                return;
            }
        }
        log_unsupported("rvv_macc_emitter_t.fmacc_float");
    }

private:
    static bool is_int(data_type_t dt) {
        using namespace data_type;
        switch (dt) {
            case s8:
            case u8:
            case s32:
            case s64: return true;
            default: return false;
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
