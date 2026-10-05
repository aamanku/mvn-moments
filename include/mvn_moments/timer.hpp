#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef MVN_ENABLE_BENCHMARKS
#define MVN_ENABLE_BENCHMARKS 0
#endif

// Usage (enable with CMake -DMVN_ENABLE_BENCHMARKS=ON):
//
//   void Factorize(mvn::Timer* timer = nullptr)
//   {
//       MVN_SCOPE(timer, "factorize");
//       // Work is timed until this function returns, including exceptions.
//   }
//
//   void Run()
//   {
//       MVN_BENCHMARK_TIMER(timer, true);
//       MVN_SCOPE(timer, "run");
//       Factorize(MVN_TIMER_ARG(timer));
//   }  // Scopes finish first; timer then prints its summary to stderr.
//
// How it works:
// - Timer owns aggregate rows and a stack of currently active scopes.
// - ScopeTimer construction pushes a start time from steady_clock and a row
//   identified by its full path, e.g. "run / factorize".
// - ScopeTimer destruction pops the frame, increments calls, and adds elapsed
//   time to total_ms. self_ms excludes time spent in nested scopes.
// - Each finished scope adds its elapsed time to its parent's child_ms. This
//   subtracts descendants exactly once when the parent computes its self time.
// - Repeated paths share a row. Rows print in first-seen order, in
//   milliseconds. Inclusive totals overlap; do not sum total_ms across
//   parents and children.
//
// Controls and lifetime:
// - Without the benchmark flag, declaration/scope macros are no-ops and do not
//   evaluate arguments; MVN_TIMER_ARG becomes nullptr. No timer is constructed
//   by the macros and no clock is read.
// - In benchmark builds, pass false to MVN_BENCHMARK_TIMER for runtime
//   disabling. MVN_SCOPE accepts a Timer reference or pointer; nullptr
//   disables that scope.
// - Use a separate Timer per thread/computation. It is not thread-safe. The
//   output stream must outlive the Timer; concurrent summaries may interleave.
// - Declare Timer before its scopes. Scopes must finish in reverse construction
//   order; ordinary block lifetimes provide this automatically.
// - Summary() can be called after all scopes finish. It prints at most once;
//   adding scopes after printing is an error. Disabled/empty timers are quiet.
// - Destruction prints automatically, including during exception unwinding.
//   Stream exceptions are suppressed in the destructor; call Summary() directly
//   if those errors must propagate. Timings include instrumentation overhead.

namespace mvn {
// One Timer per computation/thread. Declare it before all its ScopeTimers.
// ScopeTimers must be destroyed in reverse construction order.
class Timer {
   public:
    explicit Timer(bool enabled = true, std::ostream& output = std::cerr)
        : enabled_(enabled && MVN_ENABLE_BENCHMARKS), output_(output)
    {
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    ~Timer() noexcept
    {
        // Destructors cannot propagate stream exceptions during stack
        // unwinding. Call Summary() explicitly if stream error propagation is
        // required.
        try {
            Summary();
        } catch (...) {
        }
    }

    // Finish all scopes before calling; normally the destructor calls this.
    void Summary()
    {
        if (!enabled_ || printed_ || rows_.empty()) {
            return;
        }
        if (!stack_.empty()) {
            throw std::logic_error(
                "Timer summary requires all scopes to finish");
        }

        output_ << "Timing summary (inclusive / self milliseconds)\n";
        for (const auto& row : rows_) {
            output_ << row.path << " | calls=" << row.calls
                    << " | total_ms=" << row.total_ms
                    << " | self_ms=" << row.self_ms
                    << " | mean_ms=" << row.total_ms / row.calls << '\n';
        }
        printed_ = true;
    }

   private:
    friend class ScopeTimer;
    using Clock = std::chrono::steady_clock;
    struct Row {
        std::string path;
        std::size_t calls = 0;
        double total_ms = 0;
        double self_ms = 0;
    };
    struct Frame {
        std::size_t row;
        Clock::time_point start;
        double child_ms = 0;
    };
    bool enabled_;
    bool printed_ = false;
    std::ostream& output_;
    std::vector<Row> rows_;
    std::vector<Frame> stack_;
};

class ScopeTimer {
   public:
    ScopeTimer(Timer& timer, const char* name) : ScopeTimer(&timer, name)
    {
    }
    ScopeTimer(Timer* timer, const char* name) : timer_(timer)
    {
        if (!timer_ || !timer_->enabled_) {
            return;
        }
        if (timer_->printed_) {
            throw std::logic_error("Cannot add scopes after Timer summary");
        }
        if (!name || !*name) {
            throw std::invalid_argument("Timing section needs a name");
        }

        std::string path =
            timer_->stack_.empty()
                ? name
                : timer_->rows_[timer_->stack_.back().row].path + " / " + name;
        std::size_t index = 0;
        const auto row_count = timer_->rows_.size();
        for (; index < row_count; ++index) {
            if (timer_->rows_[index].path == path) {
                break;
            }
        }
        if (index == timer_->rows_.size()) {
            timer_->rows_.push_back({std::move(path)});
        }

        // Start after resolving the aggregate row for this nested path.
        timer_->stack_.push_back({index, Timer::Clock::now()});
    }
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    ~ScopeTimer() noexcept
    {
        if (!timer_ || !timer_->enabled_) {
            return;
        }

        const auto end = Timer::Clock::now();
        const auto frame = timer_->stack_.back();
        timer_->stack_.pop_back();
        const double ms =
            std::chrono::duration<double, std::milli>(end - frame.start)
                .count();

        auto& row = timer_->rows_[frame.row];
        ++row.calls;
        row.total_ms += ms;
        // Child time includes descendants, so subtract direct children once.
        row.self_ms += ms - frame.child_ms;
        if (!timer_->stack_.empty()) {
            timer_->stack_.back().child_ms += ms;
        }
    }

   private:
    Timer* timer_;
};
}  // namespace mvn

#define MVN_DETAIL_JOIN_(a, b) a##b
#define MVN_DETAIL_JOIN(a, b) MVN_DETAIL_JOIN_(a, b)
#if MVN_ENABLE_BENCHMARKS
#define MVN_BENCHMARK_TIMER(timer, enabled) ::mvn::Timer timer(enabled)
#define MVN_TIMER_ARG(timer) (&(timer))
#define MVN_SCOPE(timer, name) \
    ::mvn::ScopeTimer MVN_DETAIL_JOIN(mvn_scope_, __LINE__)(timer, name)
#else
#define MVN_BENCHMARK_TIMER(timer, enabled) ((void)0)
#define MVN_TIMER_ARG(timer) nullptr
#define MVN_SCOPE(timer, name) ((void)0)
#endif

#endif  // TIMER_HPP_
