/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "utils/timer.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>

namespace timer {

double ms_now() {
    auto timePointTmp
            = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

// TODO: remove me
#if !defined(BENCHDNN_USE_RDPMC) || defined(_WIN32)
#else
#endif

void timer_t::restart() {
    ms_.clear();
    total_ms_ = 0;
    is_finalized_ = false;
    start();
}

void timer_t::start() {
    ms_start_ = ms_now();
}

void timer_t::stop(int append_n_times) {
    stop(append_n_times, ms_now() - ms_start_);
}

void timer_t::stop(int append_n_times, double append_ms) {
    assert(!is_finalized_);
    if (is_finalized_) return;

    if (append_n_times <= 0) {
        // No measurements happened.
        return;
    }

    if (append_n_times == 1) {
        ms_.push_back(append_ms);
        total_ms_ += append_ms;
        return;
    }

    for (int i = 0; i < append_n_times; i++)
        ms_.push_back(append_ms / append_n_times);
    total_ms_ += append_ms;
}

void timer_t::finalize_results(double fast_outlier_percent_to_drop) {
    // One call is enough.
    assert(!is_finalized_);

    std::sort(ms_.begin(), ms_.end());

    if (fast_outlier_percent_to_drop * n_times() > 0) {
        int n_fast_drop
                = static_cast<int>(fast_outlier_percent_to_drop * n_times());
        const auto it = ms_.begin();
        auto it_plus_n = it;
        for (int i = 0; i < n_fast_drop; i++) {
            total_ms_ -= ms_[i];
            it_plus_n++;
        }
        ms_.erase(it, it_plus_n);
    }
    is_finalized_ = true;
}

int timer_t::n_times() const {
    return static_cast<int>(ms_.size());
}

double timer_t::ms(mode_t mode) const {
    assert(is_finalized_);
    if (!is_finalized_) { return 0; }

    switch (mode) {
        case mode_t::sum: return total_ms_;
        case mode_t::avg: return total_ms_ / n_times();
        case mode_t::min: return ms_.front();
        case mode_t::max: return ms_.back();
        default: assert(!"unknown mode");
    }
    return 0;
}

double timer_t::sec(mode_t mode) const {
    return ms(mode) / 1e3;
}

timer_t &timer_t::operator+=(const timer_t &other) {
    // If timer is finalized, can't append anything to it.
    assert(!this->is_finalized_);
    if (this->is_finalized_) return *this;

    this->total_ms_ += other.total_ms_;
    this->ms_.insert(this->ms_.end(), other.ms_.begin(), other.ms_.end());
    return *this;
}

void timer_t::dump() const {
    BENCHDNN_PRINT(0, "%s", "[TIMER]: {");
    for (auto ms : ms_) {
        BENCHDNN_PRINT(0, "%g ", ms);
    }
    BENCHDNN_PRINT(0, "}, total:%g\n", total_ms_);
}

timer_t &timer_map_t::get_timer(const std::string &name) {
    auto it = timers.find(name);
    if (it != timers.end()) return it->second;
    // Set a new timer if requested one wasn't found
    auto res = timers.emplace(name, timer_t());
    return res.first->second;
}

const std::vector<service_timers_entry_t> &get_global_service_timers() {
    // `service_timers_entry_t` type for each entry is needed for old GCC 4.8.5,
    // otherwise, it reports "error: converting to ‘std::tuple<...>’ from
    // initializer list would use explicit constructor
    // ‘constexpr std::tuple<...>’.
    static const std::vector<service_timers_entry_t> global_service_timers = {
            service_timers_entry_t {
                    "create_pd", mode_bit_t::init, timer::names::cpd_timer},
            service_timers_entry_t {
                    "create_prim", mode_bit_t::init, timer::names::cp_timer},
            service_timers_entry_t {
                    "fill", mode_bit_t::exec, timer::names::fill_timer},
            service_timers_entry_t {
                    "execute", mode_bit_t::exec, timer::names::execute_timer},
            service_timers_entry_t {
                    "compute_ref", mode_bit_t::corr, timer::names::ref_timer},
            service_timers_entry_t {
                    "compare", mode_bit_t::corr, timer::names::compare_timer},
    };
    return global_service_timers;
}

} // namespace timer
