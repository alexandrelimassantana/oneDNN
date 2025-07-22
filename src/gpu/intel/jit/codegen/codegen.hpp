/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#ifndef GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP
#define GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP

#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include <sycl/sycl.hpp>
#define WITH_SYCL_RUNTIME
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
#define WITH_OPENCL_RUNTIME
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_L0
#include "level_zero/ze_api.h"
#define WITH_L0_RUNTIME
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

#ifdef WITH_SYCL_RUNTIME
::sycl::kernel make_kernel(const kernel::iface_t &iface, const stmt_t &body,
        const kernel::options_t &options, const ngen::DebugConfig &debug_cfg,
        ::sycl::context ctx, ::sycl::device dev);
#endif
#ifdef WITH_OPENCL_RUNTIME
cl_kernel make_kernel(const kernel::iface_t &iface, const stmt_t &body,
        const kernel::options_t &options, const ngen::DebugConfig &debug_cfg,
        cl_context ctx, cl_device_id dev);
#endif
#ifdef WITH_L0_RUNTIME
std::pair<ze_module_handle_t, ze_kernel_handle_t> make_kernel(
        const kernel::iface_t &iface, const stmt_t &body,
        const kernel::options_t &options, const ngen::DebugConfig &debug_cfg,
        ze_context_handle_t ctx, ze_device_handle_t dev);
#endif

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
