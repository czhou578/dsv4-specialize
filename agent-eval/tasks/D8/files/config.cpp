#include "config.h"
#include <sstream>

static std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::map<std::string, std::string> parse_config(const std::string& text) {
    std::map<std::string, std::string> out;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        line = trim(line.substr(0, line.find('#')));
        auto eq = line.find('=');
        if (eq != std::string::npos) out[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return out;
}
