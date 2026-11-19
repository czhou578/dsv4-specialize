#pragma once
#include <cstddef>
#include <vector>

long mean(const std::vector<long>& v);  // truncates toward zero; 0 for an empty vector
std::vector<long> window_means(const std::vector<long>& v, std::size_t w);
