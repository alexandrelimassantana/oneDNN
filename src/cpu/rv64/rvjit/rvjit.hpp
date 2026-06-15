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

#ifndef CPU_RV64_RVJIT_RVJIT_HPP
#define CPU_RV64_RVJIT_RVJIT_HPP

#include <memory>

#include "common/utils.hpp"
#include "cpu/rv64/rvjit/rvjit_const_arithmetic.hpp"
#include "cpu/rv64/rvjit/rvjit_control_flow.hpp"
#include "cpu/rv64/rvjit/rvjit_emitter.hpp"
#include "cpu/rv64/rvjit/rvjit_fma.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"
#include "cpu/rv64/rvjit/rvjit_memory_move.hpp"
#include "cpu/rv64/rvjit/rvjit_register_pool.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Component-based system for Risc-V Just-In-Time code generation
class rvjit_t {
public:
    explicit rvjit_t(codegen_t &cg) : emitter_(cg) {}

    const emitter_t &emitter() const { return emitter_; }

    const_arithmetic_t &const_arithmetic() {
        if (!const_arithmetic_)
            const_arithmetic_
                    = utils::make_unique<const_arithmetic_t>(emitter_);
        return *const_arithmetic_;
    }

    control_flow_t &control_flow() {
        if (!control_flow_)
            control_flow_ = utils::make_unique<control_flow_t>(
                    emitter_, const_arithmetic());
        return *control_flow_;
    }

    register_pool_t &register_pool() {
        if (!register_pool_)
            register_pool_ = utils::make_unique<register_pool_t>(emitter_);
        return *register_pool_;
    }

    memory_move_t &memory_move() {
        if (!mem_move_) mem_move_ = utils::make_unique<memory_move_t>(emitter_);
        return *mem_move_;
    }

    rvv_macc_emitter_t &macc() {
        if (!macc_) macc_ = utils::make_unique<rvv_macc_emitter_t>(emitter_);
        return *macc_;
    }

#if XBYAK_RISCV_V
    rvv_matmul_t &matmul() {
        if (!matmul_)
            matmul_ = utils::make_unique<rvv_matmul_t>(emitter_,
                    register_pool(), memory_move(), macc(), const_arithmetic(),
                    control_flow());
        return *matmul_;
    }
#endif

private:
    emitter_t emitter_;
    std::unique_ptr<const_arithmetic_t> const_arithmetic_;
    std::unique_ptr<control_flow_t> control_flow_;
    std::unique_ptr<register_pool_t> register_pool_;
    std::unique_ptr<memory_move_t> mem_move_;
    std::unique_ptr<rvv_macc_emitter_t> macc_;
#if XBYAK_RISCV_V
    std::unique_ptr<rvv_matmul_t> matmul_;
#endif
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
