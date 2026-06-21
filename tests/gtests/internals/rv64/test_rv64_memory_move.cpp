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
#include <cstring>
#include <tuple>
#include <vector>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

// Invalid combinations: all emit nothing, leaving only ret().

struct move_noop_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(move_noop_probe_t)
    move_noop_probe_t() : jit_generator_t("rv64_macc_vv_noop_probe") {}

protected:
    void generate() override {
        using namespace data_type;

        rvjit_t m(*this);
        auto &mem = m.memory_move();

        // None of these should emit anything
        mem.fload(ft0, t0, u8, 0); // Invalid float type
        mem.xload(t0, t0, f16, 0); // Invalid int type
        mem.vload(v0, vaddr_t::unit(t0), f8_e5m2); // Invalid vector type
        mem.vstore(v0, vaddr_t::unit(t0), f8_e5m2); // Invalid vector type

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(MoveNoop, InvalidParametersEmitNothing) {
    move_noop_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    ASSERT_EQ(k.getSize(), sizeof(uint32_t));
}

// Valid combinations: probe and reference instructions interleaved

struct move_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(move_probe_t)
    move_probe_t() : jit_generator_t("rv64_macc_vv_valid_probe") {}

protected:
    void generate() override {
        using namespace data_type;

        rvjit_t m(*this);
        auto &mem = m.memory_move();

        // Float loads
        mem.fload(ft0, t0, f16, 0);
        flh(ft0, t0, 0);
        mem.fload(ft0, t0, f32, 0);
        flw(ft0, t0, 0);
        mem.fload(ft0, t0, f64, 0);
        fld(ft0, t0, 0);
        mem.fload(ft0, t0, bf16, 0);
        flh(ft0, t0, 0);

        // Integer loads
        mem.xload(t0, t0, u8, 0);
        lbu(t0, t0, 0);
        mem.xload(t0, t0, s8, 0);
        lb(t0, t0, 0);
        mem.xload(t0, t0, s32, 0);
        lw(t0, t0, 0);
        mem.xload(t0, t0, s64, 0);
        ld(t0, t0, 0);

        // Vector unit-stride loads
        mem.vload(v0, vaddr_t::unit(t0), u8);
        vle8_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), s8);
        vle8_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), s32);
        vle32_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), s64);
        vle64_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), f16);
        vle64_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), f32);
        vle32_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), f64);
        vle64_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), bf16);
        vle16_v(v0, t0);

        // Vector unit-stride stores
        mem.vstore(v0, vaddr_t::unit(t0), u8);
        vse8_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), s8);
        vse8_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), s32);
        vse32_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), s64);
        vse64_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), f16);
        vse64_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), f32);
        vse32_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), f64);
        vse64_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), bf16);
        vse16_v(v0, t0);

        // Vector constant stride loads
        mem.vload(v0, vaddr_t::strided(t0, t1), u8);
        vlse8_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), s8);
        vlse8_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), s32);
        vlse32_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), s64);
        vlse64_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), f16);
        vlse64_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), f32);
        vlse32_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), f64);
        vlse64_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), bf16);
        vlse16_v(v0, t0, t1);

        // Vector constant stride stores
        mem.vstore(v0, vaddr_t::strided(t0, t1), u8);
        vsse8_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), s8);
        vsse8_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), s32);
        vsse32_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), s64);
        vsse64_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), f16);
        vsse64_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), f32);
        vsse32_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), f64);
        vsse64_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), bf16);
        vsse16_v(v0, t0, t1);

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(MoveOp, ValidParametersEmitExpectedInstructions) {
    move_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    constexpr int n_pairs = 4;
    const auto *code = k.getCode<const uint32_t *>();
    ASSERT_GT(k.getSize(), n_pairs * 2 * sizeof(uint32_t));
    for (int i = 0; i < n_pairs; ++i)
        EXPECT_EQ(code[2 * i], code[2 * i + 1]) << "mismatch at pair " << i;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
