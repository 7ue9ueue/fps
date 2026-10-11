// fps::detail::Words: the frontend's storage for words modulo P (Polynomial's coefficients,
// Factorials' tables).
//
//   detail::Words w(n);                                // n zeros
//   detail::Words w(n, detail::Words::Unwritten{});    // n words, unwritten: the caller fills them
//   w.size(), w.data(), w.resize(n)                    // resize zero-fills past the old size
//
// Unlike std::vector it can allocate without writing, so an operator writes its result in one
// pass, and it copies with memcpy (std::vector with such an allocator copies word by word under
// GCC 15). With no spare capacity, growing one word at a time costs O(n^2), and copy assignment
// always allocates.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace fps::detail {

class Words {
public:
    struct Unwritten {};

    Words() = default;
    explicit Words(std::size_t n) : Words(n, Unwritten{}) { std::fill_n(data(), n, 0u); }
    Words(std::size_t n, Unwritten)
        : words_(n ? std::make_unique_for_overwrite<std::uint32_t[]>(n) : nullptr), size_(n) {}
    Words(const Words& w) : Words(w.size_, Unwritten{}) { std::copy_n(w.data(), size_, data()); }
    Words(Words&& w) noexcept : words_(std::move(w.words_)), size_(std::exchange(w.size_, 0)) {}
    Words& operator=(const Words& w) { return *this = Words(w); }
    Words& operator=(Words&& w) noexcept {
        words_ = std::move(w.words_);
        size_ = std::exchange(w.size_, 0);
        return *this;
    }

    std::size_t size() const { return size_; }
    std::uint32_t* data() { return words_.get(); }
    const std::uint32_t* data() const { return words_.get(); }

    // Words past the old size are zero.
    void resize(std::size_t n) {
        if (n <= size_) return void(size_ = n);
        Words w(n, Unwritten{});
        std::copy_n(data(), size_, w.data());
        std::fill(w.data() + size_, w.data() + n, 0u);
        *this = std::move(w);
    }

private:
    std::unique_ptr<std::uint32_t[]> words_;
    std::size_t size_ = 0;
};

}  // namespace fps::detail
