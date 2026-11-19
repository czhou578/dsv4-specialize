#include "brackets.h"

int bracket(const std::vector<long>& thr, long x) {
    int lo = 0, hi = static_cast<int>(thr.size());
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (thr[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}
