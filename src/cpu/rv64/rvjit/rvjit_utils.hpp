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

#ifndef CPU_RV64_RVJIT_RVJIT_UTILS_HPP
#define CPU_RV64_RVJIT_RVJIT_UTILS_HPP

#include "common/type_helpers.hpp"

#include "cpu/rv64/rvjit/rvjit_emitter.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Determines if an integer value is representable as a signed 12-bit value
bool is_simm12(int i);

/// Determines if a general-purpose register can be written to
bool is_writeable(const Reg &r);

/// Maps a narrow oneDNN data type to its natural 2x-wide counterpart
data_type_t natural_wide(const data_type_t &dt);

/// Determines if an oneDNN integer data type is signed
bool is_signed_int(const data_type_t &dt);

/// Obtains the RVV LMUL value for a given group size
/// @details defaults to m1 for group_sizes other than 2, 4, and 8.
LMUL lmul_for(const int group_size);

/// Obtain the number of vector registers in a group for a given lmul setting
int vgroup_size(const LMUL &m);

/// Obtain the number of vector register groups for a given lmul setting
int get_nvgroups(const LMUL &m);

/// Single-element-width size in bytes for a given oneDNN data type
int sewb(const data_type_t &dt);

/// Obtains the RVV SEW value for a given oneDNN data type
SEW sew_for(const data_type_t &dt);

/// Single-element-width size in bytes for a given sew setting
int sewb(const SEW &s);

/// Computes the maximum vector length given a certain VPU configuration
int get_maximum_vector_length(
        const int vlenb, const SEW &sew = SEW::e8, const LMUL &lmul = LMUL::m1);

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
