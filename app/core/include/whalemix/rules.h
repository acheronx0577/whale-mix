#pragma once

#include <string>
#include <utility>

namespace whalemix::rules {

inline constexpr double kLayaMinProbability = 0.6;

std::string sound_label(double bat, double baf);

std::pair<std::string, std::string> mix_label(
    const std::string& sound,
    const std::string& laya_choice,
    double laya_probability,
    double min_probability = kLayaMinProbability
);

} // namespace whalemix::rules
