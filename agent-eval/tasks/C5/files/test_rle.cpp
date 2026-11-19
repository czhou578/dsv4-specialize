#include "rle.h"
#include "check.h"

int main() {
    CHECK(rle_encode("aaab") == "3a1b");
    CHECK(rle_decode("3a1b") == "aaab");
    CHECK(rle_encode(std::string(12, 'x')) == "12x");
    THROWS(rle_decode("a"), std::invalid_argument);
    std::puts("ok");
}
