#pragma once
#include <cctype>
#include <stdexcept>
#include <string>

inline bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)); }

inline std::string rle_encode(const std::string& s) {
    std::string out;
    for (std::size_t i = 0, j; i < s.size(); i = j) {
        if (is_digit(s[i])) throw std::invalid_argument("digit");
        for (j = i; j < s.size() && s[j] == s[i]; j++) {}
        out += std::to_string(j - i) + s[i];
    }
    return out;
}

inline std::string rle_decode(const std::string& s) {
    std::string out;
    for (std::size_t i = 0, j; i < s.size(); i = j + 1) {
        for (j = i; j < s.size() && is_digit(s[j]); j++) {}
        if (j == i || j == s.size() || s[i] == '0') throw std::invalid_argument("bad run");
        out.append(std::stoul(s.substr(i, j - i)), s[j]);
    }
    return out;
}
