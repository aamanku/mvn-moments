#ifndef LOG_HPP_
#define LOG_HPP_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string_view>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace mvn {
enum class Color { kAutomatic, kAlways, kNever };
enum class Level { kInfo, kWarning, kError };

inline bool StderrIsTerminal()
{
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}

// Diagnostics go to stderr. Logging never substitutes for returning a Status.
inline void Log(Level level, std::string_view message,
                Color color = Color::kAutomatic)
{
    const bool colored = color == Color::kAlways ||
                         (color == Color::kAutomatic &&
                          !std::getenv("NO_COLOR") && StderrIsTerminal());
    const char* label = level == Level::kError     ? "error"
                        : level == Level::kWarning ? "warning"
                                                   : "info";
    const char* ansi = level == Level::kError     ? "\033[31m"
                       : level == Level::kWarning ? "\033[33m"
                                                  : "\033[36m";
    std::cerr << (colored ? ansi : "") << '[' << label << ']'
              << (colored ? "\033[0m" : "") << ' ' << message << '\n';
}
}  // namespace mvn

#endif  // LOG_HPP_
