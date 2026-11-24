#include "whalemix/rules.h"

namespace whalemix::rules {

std::string sound_label(double bat, double baf) {
    return (bat > baf) ? "call" : "not-call";
}

std::pair<std::string, std::string> mix_label(
    const std::string& sound,
    const std::string& laya_choice,
    double laya_probability,
    double min_probability
) {
    std::string s = sound.empty() ? "not-call" : sound;
    if (laya_probability < min_probability || laya_choice.empty()) {
        return {s, "blue-a"};
    }
    if (laya_choice == s) {
        return {s, "agree"};
    }
    return {"review", "disagree"};
}

} // namespace whalemix::rules
