#include <sstream>
#include <string>

#include "check.hpp"
#include "mvn_moments/moments.hpp"

namespace {
using check::Require;

void Helper(mvn::Timer* timer)
{
    MVN_SCOPE(timer, "helper");
    (void)timer;
}

void TestOrder()
{
    static_assert((mvn::Order::kZeroth | mvn::Order::kSecond) ==
                  mvn::Order::kZerothSecond);
    static_assert(mvn::HasOrder(mvn::Order::kAll, mvn::Order::kFirstSecond));
    static_assert(!mvn::HasOrder(mvn::Order::kFirst, mvn::Order::kSecond));
}

void TestTimer()
{
    std::ostringstream output;
    {
        mvn::Timer timer(true, output);
        {
            mvn::ScopeTimer outer(timer, "outer");
            for (int i = 0; i < 2; ++i) {
                mvn::ScopeTimer inner(timer, "inner");
            }
        }
        timer.Summary();  // Destructor must not print a second summary.
    }

#if MVN_ENABLE_BENCHMARKS
    Require(output.str().find("outer / inner | calls=2") != std::string::npos,
            "Nested scopes must aggregate by path");
    Require(output.str().find("Timing summary") ==
                output.str().rfind("Timing summary"),
            "Summary must print once");
#else
    Require(output.str().empty(),
            "Disabled benchmarking must produce no output");
#endif
}

void TestMacros()
{
    int evaluated = 0;
    MVN_BENCHMARK_TIMER(optional_timer, ++evaluated);
    {
        MVN_SCOPE(optional_timer, "macro_scope");
        Helper(MVN_TIMER_ARG(optional_timer));
    }

#if !MVN_ENABLE_BENCHMARKS
    Require(evaluated == 0, "Disabled macros must not evaluate arguments");
#endif
}
}  // namespace

int main()
{
    TestOrder();
    TestTimer();
    TestMacros();
}
