#include <cstdio>
#include "brackets.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(parcel_fee(120) == 499);
    CHECK(parcel_fee(1500) == 899);
    CHECK(order_fee({300, 500, 2000}) == 499 + 899 + 1599);
    std::printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
    return fails != 0;
}
