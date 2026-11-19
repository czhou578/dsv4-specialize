#include "rle.h"
#include "check.h"

int main() {
    CHECK(rle_encode("") == "" && rle_decode("") == "");
    CHECK(rle_encode("  b") == "2 1b" && rle_decode("2 1b") == "  b");
    CHECK(rle_encode("abba") == "1a2b1a");
    CHECK(rle_decode("10x") == std::string(10, 'x'));
    CHECK(rle_decode("2a3a") == "aaaaa");
    THROWS(rle_encode("a1"), std::invalid_argument);
    THROWS(rle_decode("3"), std::invalid_argument);
    THROWS(rle_decode("3a2"), std::invalid_argument);
    THROWS(rle_decode("0a"), std::invalid_argument);
    THROWS(rle_decode("03a"), std::invalid_argument);
    THROWS(rle_decode("1a0b"), std::invalid_argument);
    std::string s = "zz yyy!!!!!!!!!!!q";
    CHECK(rle_decode(rle_encode(s)) == s);
    std::puts("ok");
}
