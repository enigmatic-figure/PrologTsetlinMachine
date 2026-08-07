#pragma once

namespace ptm {

struct FredkinResult {
    bool control;
    bool first;
    bool second;

    friend constexpr bool operator==(const FredkinResult&,
                                     const FredkinResult&) = default;
};

[[nodiscard]] constexpr FredkinResult fredkin_gate(bool control,
                                                   bool first,
                                                   bool second) noexcept {
    return control ? FredkinResult{control, second, first}
                   : FredkinResult{control, first, second};
}

// Routes literal_truth to `first` for Include and the conjunction identity
// true to `first` for Exclude. `second` is retained as the garbage line.
[[nodiscard]] constexpr FredkinResult fredkin_literal_condition(
    bool action_include,
    bool literal_truth) noexcept {
    return fredkin_gate(action_include, true, literal_truth);
}

}  // namespace ptm

