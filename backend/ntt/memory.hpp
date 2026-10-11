// Memory of ntt::Convolution, kept per thread across calls (issue #38). Mapping fresh memory per
// product cost 56 us for a 32 x 32 product on Zen 3, against 0.11 us for the product itself: the
// first touch of a fresh huge page (notes.md).
//
// Regions: mappings of whole 2 MiB pages, 2 MiB aligned, in huge pages where the system allows.
// Each thread keeps one free region, the largest given back; take_region() hands it out if it is
// large enough. Twiddle tables past 2^kStaticLog: one per thread, rebuilt when a longer one is
// needed. A thread's region and tables are unmapped when it exits.
#pragma once

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "backend/ntt/common.hpp"

namespace fps::ntt::detail {

inline constexpr std::size_t kHugePage = std::size_t(1) << 21;

struct Region {
    std::byte* start = nullptr;  // 2 MiB aligned; null if empty
    std::size_t bytes = 0;       // a multiple of 2 MiB
};

inline Region map_region(std::size_t bytes) {
    bytes = (bytes + kHugePage - 1) / kHugePage * kHugePage;
    void* p = ::mmap(nullptr, bytes + kHugePage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    auto* raw = static_cast<std::byte*>(p);
    const std::size_t head = (kHugePage - reinterpret_cast<std::uintptr_t>(raw) % kHugePage) % kHugePage;
    if (head) ::munmap(raw, head);
    ::munmap(raw + head + bytes, kHugePage - head);
#ifdef MADV_HUGEPAGE
    ::madvise(raw + head, bytes, MADV_HUGEPAGE);
#endif
    return {raw + head, bytes};
}

inline void unmap_region(Region r) {
    if (r.start) ::munmap(r.start, r.bytes);
}

// Per-thread state, trivially destructible so it stays usable after ThreadMemoryCloser has run
// (in the destructors of static objects at exit): closed then means "keep no region".
struct ThreadMemory {
    Region kept;                 // a free region
    Region tables;               // forward table, then inverse table; empty or for 2^tables_lg
    int tables_lg = 0;
    bool closed = false;
};
inline thread_local ThreadMemory thread_memory;

struct ThreadMemoryCloser {
    ~ThreadMemoryCloser() {
        unmap_region(std::exchange(thread_memory.kept, {}));
        unmap_region(std::exchange(thread_memory.tables, {}));
        thread_memory.tables_lg = 0;
        thread_memory.closed = true;
    }
};

// Registers the closer of this thread on first use. Not after it has run: passing the declaration
// of a destroyed block variable is undefined.
inline void open_thread_memory() { static thread_local ThreadMemoryCloser closer; }

// A region of at least `bytes` bytes: the kept one if it is large enough, else a new mapping.
inline Region take_region(std::size_t bytes) {
    if (thread_memory.kept.bytes >= bytes) return std::exchange(thread_memory.kept, {});
    return map_region(bytes);
}

// Keeps r if it is larger than the kept region, unmapping the smaller one.
inline void give_region(Region r) {
    ThreadMemory& m = thread_memory;
    if (m.closed || r.bytes <= m.kept.bytes) return unmap_region(r);
    open_thread_memory();
    unmap_region(std::exchange(m.kept, r));
}

// The forward and inverse tables of this thread, entries [0, 2^lg / 16) at least, built by Variant
// if they are shorter. lg > kStaticLog. Valid until the next call on this thread.
template <class Variant>
std::pair<const std::uint32_t*, const std::uint32_t*> thread_tables(int lg) {
    ThreadMemory& m = thread_memory;
    if (m.tables_lg < lg) {
        unmap_region(std::exchange(m.tables, {}));
        m.tables_lg = 0;
        const std::size_t words = table_words(lg);
        Region r = map_region(2 * words * sizeof(std::uint32_t));
        auto* t = reinterpret_cast<std::uint32_t*>(r.start);
        Variant::build_table(t, (std::size_t(1) << lg) / 16, false);
        Variant::build_table(t + words, (std::size_t(1) << lg) / 16, true);
        if (!m.closed) open_thread_memory();  // once closed (exit), tables stay until the process ends
        m.tables = r, m.tables_lg = lg;
    }
    const auto* t = reinterpret_cast<const std::uint32_t*>(m.tables.start);
    return {t, t + table_words(m.tables_lg)};
}

}  // namespace fps::ntt::detail
