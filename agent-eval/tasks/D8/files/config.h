#pragma once
#include <map>
#include <string>

// Parses "key = value" lines; '#' starts a comment; a later key replaces an earlier one.
std::map<std::string, std::string> parse_config(const std::string& text);
