#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) : buf_(capacity) {
        if (capacity == 0) throw std::invalid_argument("capacity 0");
    }
    void push(int value) {
        if (n_ == buf_.size()) pop();
        buf_[(head_ + n_++) % buf_.size()] = value;
    }
    int pop() {
        int v = at(0);
        head_ = (head_ + 1) % buf_.size();
        n_--;
        return v;
    }
    int at(std::size_t i) const {
        if (i >= n_) throw std::out_of_range("index");
        return buf_[(head_ + i) % buf_.size()];
    }
    std::size_t size() const { return n_; }
    std::size_t capacity() const { return buf_.size(); }
private:
    std::vector<int> buf_;
    std::size_t head_ = 0, n_ = 0;
};
