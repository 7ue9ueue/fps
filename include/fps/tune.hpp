// Tuning: the backend's choices (variants, cutoffs) measured on this machine, and kept in a file.
//
//   fps::tune_all()             // measure every choice below on this machine; once per process
//   fps::save_wisdom(path)      // write the active choices to path; false if it cannot be written
//   fps::load_wisdom(path)      // set the choices path names; false, setting nothing, if path is
//                               // missing, malformed, or was saved on another CPU or target
//
//   if (!fps::load_wisdom(path)) {   // the usual use: tune once per machine
//       fps::tune_all();
//       fps::save_wisdom(path);
//   }
//
// Without these, every call uses the compiled defaults, measured per target on our machines
// (tools/tune.py). Tuning pays most on another CPU: up to 1.6x at some sizes
// (backend/tune/notes.md). A choice changes speed only, never a result. Nothing tunes unless
// asked: no lazy tuning on first use, so a call costs the same every time.
//
// tune_all() takes about 0.1 s (frontend/notes.md). It measures afresh, from the compiled defaults,
// and replaces what load_wisdom() set; a second call returns at once. Wisdom is text: the format,
// the CPU, the target, then one line per choice. load_wisdom() checks every line before it sets
// any; choices the file does not name keep their values, and lines for choices this build lacks
// are skipped. Neither function throws on a bad path. All three are safe while other threads
// compute: those see the old choices or the new.
//
// -DFPS_NO_TUNER (e.g. single-file contest bundles) compiles the defaults in as constants and the
// tuner out: tune_all() does nothing, save_wisdom() writes nothing and returns false, and
// load_wisdom() reads nothing and returns false, so the code above still compiles and runs.
//
// The name is not fps::tune(), as DESIGN.md first had it: fps::tune is the backend's namespace.
#pragma once

#include <string>

// A kernel's choices register only if its tune header is included. Every kernel the frontend
// calls that has tuned choices includes its backend/<kernel>/tune.hpp here (ntt's when issue #78
// adds one), so tune_all() and the wisdom cover them all.
#ifndef FPS_NO_TUNER
#include "backend/factorials/tune.hpp"
#include "backend/small/tune.hpp"
#include "backend/tune/tune.hpp"
#endif

namespace fps {

#ifndef FPS_NO_TUNER

inline void tune_all() { tune::run(); }
inline bool save_wisdom(const std::string& path) { return tune::save(path); }
inline bool load_wisdom(const std::string& path) { return tune::load(path); }

#else

inline void tune_all() {}
inline bool save_wisdom(const std::string&) { return false; }
inline bool load_wisdom(const std::string&) { return false; }

#endif

}  // namespace fps
