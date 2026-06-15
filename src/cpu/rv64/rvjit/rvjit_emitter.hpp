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

#ifndef CPU_RV64_RVJIT_RVJIT_EMITTER_HPP
#define CPU_RV64_RVJIT_RVJIT_EMITTER_HPP

#include <functional>

#ifndef XBYAK_RISCV_V
#define XBYAK_RISCV_V 1
#endif

#include "xbyak_riscv/xbyak_riscv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

using Reg = Xbyak_riscv::Reg;
using FReg = Xbyak_riscv::FReg;
using VReg = Xbyak_riscv::VReg;
using Label = Xbyak_riscv::Label;
using LMUL = Xbyak_riscv::LMUL;
using SEW = Xbyak_riscv::SEW;
using VM = Xbyak_riscv::VM;
using VTA = Xbyak_riscv::VTA;
using VMA = Xbyak_riscv::VMA;
using codegen_t = Xbyak_riscv::CodeGenerator;

/// Callback type to emit code for a parametrized code block
template <typename... Args>
using callback_t = std::function<void(Args...)>;

// Callback type to emit code for a constant basic block
using on_enter_t = callback_t<>;

/// Non-owning handle to the underlying code generator used by the components
class emitter_t {
public:
    explicit emitter_t(codegen_t &cg) : cg_(&cg) {}

    codegen_t *operator->() const { return cg_; }
    codegen_t &cg() const { return *cg_; }

private:
    codegen_t *cg_;
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
