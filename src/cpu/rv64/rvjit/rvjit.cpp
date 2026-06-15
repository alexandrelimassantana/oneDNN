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

#include "cpu/rv64/rvjit/rvjit_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Determines if an integer value is representable as a signed 12-bit value
bool is_simm12(int i) {
    return -2048 <= i && i < 2048;
}

/// Determines if a general-purpose register can be written to
bool is_writeable(const Reg &r) {
    return r != Xbyak_riscv::x0;
}

/// Maps a narrow oneDNN data type to its natural 2x-wide counterpart
data_type_t natural_wide(const data_type_t &dt) {
    using namespace data_type;
    switch (dt) {
        case f16:
        case bf16: return f32;
        case f32: return f64;
        case s32: return s64;
        default: return data_type::undef;
    }
}

/// Determines if an oneDNN integer data type is signed
bool is_signed_int(const data_type_t &dt) {
    using namespace data_type;
    switch (dt) {
        case s8:
        case s32:
        case s64: return true;
        default: return false;
    }
}

/// Obtains the RVV LMUL value for a given group size
/// @details defaults to m1 for group_sizes other than 2, 4, and 8.
LMUL lmul_for(const int group_size) {
    switch (group_size) {
        case 1: return LMUL::m1;
        case 2: return LMUL::m2;
        case 4: return LMUL::m4;
        case 8: return LMUL::m8;
        default: return LMUL::m1;
    }
}

/// Obtain the number of vector registers in a group for a given lmul setting
int vgroup_size(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 1;
        case LMUL::m2: return 2;
        case LMUL::m4: return 4;
        case LMUL::m8: return 8;
        default: return 1;
    }
}

/// Obtain the number of vector register groups for a given lmul setting
int get_nvgroups(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 32;
        case LMUL::m2: return 16;
        case LMUL::m4: return 8;
        case LMUL::m8: return 4;
        default: return 1;
    }
}

/// Single-element-width size in bytes for a given oneDNN data type
int sewb(const data_type_t &dt) {
    return types::data_type_size(dt);
}

/// Obtains the RVV SEW value for a given oneDNN data type
SEW sew_for(const data_type_t &dt) {
    switch (sewb(dt)) {
        case 8: return SEW::e64;
        case 4: return SEW::e32;
        case 2: return SEW::e16;
        case 1: return SEW::e8;
        default: return SEW::e8;
    }
}

/// Single-element-width size in bytes for a given sew setting
int sewb(const SEW &s) {
    switch (s) {
        case SEW::e8: return 1;
        case SEW::e16: return 2;
        case SEW::e32: return 4;
        case SEW::e64: return 8;
        default: return 0;
    }
}

/// Computes the maximum vector length given a certain VPU configuration
int get_maximum_vector_length(const int vlb, const SEW &s, const LMUL &m) {
    return vlb * vgroup_size(m) / sewb(s);
}

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
