#include <cstdio>
#include "../brackets.h"
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main() {
    CHECK(bracket({0, 10, 20}, 0) == 0);
    CHECK(bracket({0, 10, 20}, 10) == 1);
    CHECK(bracket({0, 10, 20}, 19) == 1);
    CHECK(bracket({0, 10, 20}, 20) == 2);
    CHECK(bracket({0, 10, 20}, 99) == 2);
    CHECK(parcel_fee(0) == 499 && parcel_fee(499) == 499 && parcel_fee(9999) == 1599);
    CHECK(parcel_fee(10000) == 3999 && parcel_fee(25000) == 3999);
    std::printf("hidden ok\n");
    return 0;
}
