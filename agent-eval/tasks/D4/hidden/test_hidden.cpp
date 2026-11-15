#include <cstdio>
#include "../stats.h"
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main() {
    CHECK(mean({-1}) == -1);
    CHECK(mean({-7, 2}) == -2);
    CHECK(mean({5, 6}) == 5);
    CHECK(mean({}) == 0);
    CHECK(mean({-4000000000L, -2000000000L}) == -3000000000L);
    CHECK(window_means({-9, 3, -6, 1}, 2) == (std::vector<long>{-3, -1, -2}));
    std::printf("hidden ok\n");
    return 0;
}
