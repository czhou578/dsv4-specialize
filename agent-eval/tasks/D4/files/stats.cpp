#include "stats.h"

long mean(const std::vector<long>& v) {
    if (v.empty()) return 0;
    long sum = 0;
    for (long x : v) sum += x;
    return sum / v.size();
}

std::vector<long> window_means(const std::vector<long>& v, std::size_t w) {
    std::vector<long> out;
    for (std::size_t i = 0; i + w <= v.size(); ++i)
        out.push_back(mean(std::vector<long>(v.begin() + i, v.begin() + i + w)));
    return out;
}
