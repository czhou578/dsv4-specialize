#include "ring.h"
#include "check.h"

int main() {
    THROWS(RingBuffer(0), std::invalid_argument);
    RingBuffer r(3);
    THROWS(r.pop(), std::out_of_range);
    for (int v = 1; v <= 5; v++) r.push(v);
    CHECK(r.size() == 3 && r.at(0) == 3 && r.at(2) == 5);
    THROWS(r.at(3), std::out_of_range);
    CHECK(r.pop() == 3);
    r.push(6);
    CHECK(r.at(0) == 4 && r.at(1) == 5 && r.at(2) == 6);
    RingBuffer w(2);
    for (int v = 0; v < 100; v++) { w.push(v); CHECK(w.pop() == v); }
    CHECK(w.size() == 0);
    THROWS(w.at(0), std::out_of_range);
    RingBuffer one(1);
    one.push(7);
    one.push(8);
    CHECK(one.size() == 1 && one.at(0) == 8 && one.pop() == 8);
    std::puts("ok");
}
