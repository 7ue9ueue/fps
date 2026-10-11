// Prefaulting fresh output memory, so a kernel's first writes take no page fault per page (Linux).
//
//   fps::factorials::prefault(words)   // true if it populated words' whole pages
//
// If `words` looks fresh (memory the allocator just mapped and nobody wrote, as Factorials' large
// tables are), one madvise(MADV_POPULATE_WRITE) maps every whole page of `words` writable,
// zero-filled, as the writes would: one system call instead of a fault per page. The data does
// not change. Fresh: a word on the last whole page reads 0 and the page before it is not resident
// (mincore; reading a fresh page maps the shared zero page there, so it would count as resident).
// A nonzero word means the memory holds data (e.g. earlier tables): one load, no system call.
// Nothing for fewer than 2 whole pages, on other systems, before Linux 5.14 (madvise fails), or
// with -DFPS_NO_PREFAULT. Costs: notes.md (Prefault).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__linux__) && !defined(FPS_NO_PREFAULT)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fps::factorials {

inline bool prefault(std::span<std::uint32_t> words) {
#if defined(__linux__) && defined(MADV_POPULATE_WRITE) && !defined(FPS_NO_PREFAULT)
    const auto page = std::uintptr_t(sysconf(_SC_PAGESIZE));
    const auto first = reinterpret_cast<std::uintptr_t>(words.data());
    const std::uintptr_t begin = (first + page - 1) / page * page;
    const std::uintptr_t end = (first + words.size_bytes()) / page * page;
    if (end < begin + 2 * page || *reinterpret_cast<const volatile std::uint32_t*>(end - page) != 0) return false;
    unsigned char resident = 1;
    if (mincore(reinterpret_cast<void*>(end - 2 * page), page, &resident) != 0 || (resident & 1)) return false;
    return madvise(reinterpret_cast<void*>(begin), end - begin, MADV_POPULATE_WRITE) == 0;
#else
    (void)words;
    return false;
#endif
}

}  // namespace fps::factorials
