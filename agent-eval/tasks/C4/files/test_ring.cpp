#include "ring.h"
#include "check.h"

int main() {
    RingBuffer r(3);
    r.push(1);
    r.push(2);
    CHECK(r.size() == 2 && r.capacity() == 3);
    CHECK(r.at(0) == 1 && r.at(1) == 2);
    CHECK(r.pop() == 1 && r.size() == 1);
    std::puts("ok");
}
