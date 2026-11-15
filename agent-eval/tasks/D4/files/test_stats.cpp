#include <cstdio>
#include "stats.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // temperatures in tenths of a degree
    CHECK(window_means({215, 230, 245, 260}, 2) == (std::vector<long>{222, 237, 252}));
    CHECK(window_means({-30, -50, -40, 10}, 3) == (std::vector<long>{-40, -26}));
    std::printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
    return fails != 0;
}
