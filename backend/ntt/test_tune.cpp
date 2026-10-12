// Tests for backend/ntt: the tuned variant with every candidate per op and length, measure(),
// wisdom.
#include "backend/ntt/test_util.hpp"

namespace {

// A table with candidate (op + lg) % kCandidates: neighbouring ops and lengths differ.
Choices mixed_choices() {
    Choices choices{};
    for (int op = 0; op < kOps; ++op)
        for (int i = 0; i < kLengths; ++i) choices[op][i] = std::uint8_t((op + i) % kCandidates);
    return choices;
}

// The tuned variant (select.hpp) with every candidate in every place, then a mixed table up to 2^17
// (tables past 2^16 are built by tuned::build_table). Each candidate is also tested on its own.
void test_tuned() {
    for (int v = 0; v < kCandidates; ++v) {
        Choices same{};
        for (auto& row : same) row.fill(std::uint8_t(v));
        variants.set(same);
        const std::string name = "tuned, all " + variant_names()[v];
        test_transforms<variant::tuned>(name.c_str(), 10);
        test_products<variant::tuned>(name.c_str(), 10);
        test_multiply_spans<variant::tuned>(name.c_str(), 10);
    }
    variants.set(mixed_choices());
    test_transforms<variant::tuned>("tuned, mixed", 17);
    test_products<variant::tuned>("tuned, mixed", 17);
    test_multiply_spans<variant::tuned>("tuned, mixed", 17);
    for (int op = 0; op < kOps; ++op)
        for (int lg = kMinLog; lg <= kMaxLog; ++lg)
            expect(choice(Op(op), lg) == (kCandidates == 1 ? 0 : (op + lg - kMinLog) % kCandidates), "tuned: choice", lg);
    variants.set(kDefaults);
    expect(valid(kDefaults) && variants.get() == kDefaults, "tuned: compiled defaults");
}

// measure() at a small budget: valid choices; past max_lg start's, or the choice at max_lg where it
// changed; a margin above any difference keeps start. The wisdom text; "ntt.variant" through
// tune::save and load, and tune::run().
void test_measure() {
    const Choices mixed = mixed_choices();
    const Budget quick = {8, 1, 2e3, 0};
    const Choices measured = measure(mixed, quick);
    expect(valid(measured), "measure: valid");
    for (int op = 0; op < kOps; ++op) {
        const int last = quick.max_lg - kMinLog;
        const bool moved = measured[op][last] != mixed[op][last];
        for (int i = last + 1; i < kLengths; ++i)
            expect(measured[op][i] == (moved ? measured[op][last] : mixed[op][i]), "measure: past max_lg", i);
    }
    expect(measure(mixed, {7, 1, 2e3, 1e9}) == mixed, "measure: a huge margin keeps start");
    std::vector<Times> log;
    measure(kDefaults, {7, 1, 2e3, 0.03}, &log);
    expect(log.size() == (kCandidates > 1 ? std::size_t(2 * kOps) : 0u), "measure: log", log.size());
    for (const Times& t : log)
        for (const auto& ns : t.ns) expect(ns.size() == std::size_t(kCandidates) && ns[0] > 0, "measure: log times");

    expect(parse(text(mixed)) == mixed, "wisdom: text round trip");
    const std::string good = text(mixed);
    expect(!parse(std::to_string(kSignature ^ 1) + good.substr(good.find(' '))), "wisdom: another list's signature");
    expect(!parse(good.substr(0, good.rfind(' '))), "wisdom: a value short");
    expect(!parse(good.substr(0, good.rfind(' ')) + " " + std::to_string(kCandidates)), "wisdom: no such candidate");
    const std::string path = "fps-ntt-test-wisdom-" + std::to_string(std::random_device{}());
    variants.set(mixed);
    expect(fps::tune::save(path), "wisdom: save");
    variants.set(kDefaults);
    expect(fps::tune::load(path) && variants.get() == mixed, "wisdom: load");
    std::remove(path.c_str());
    variants.set(kDefaults);
    fps::tune::run();
    expect(valid(variants.get()), "tune::run: valid");
    variants.set(kDefaults);
}

}  // namespace

int main() {
    test_tuned();
    test_measure();
    return finish();
}
