#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) { throw std::logic_error("not implemented"); }
    void push(int value) {}
    int pop() { return 0; }
    int at(std::size_t i) const { return 0; }
    std::size_t size() const { return 0; }
    std::size_t capacity() const { return 0; }
};
