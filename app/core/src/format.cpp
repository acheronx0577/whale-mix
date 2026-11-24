#include "whalemix/format.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace whalemix::format {

namespace {

std::string format_2f(double val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", val);
    return std::string(buf);
}

double parse_prob(const std::string& str) {
    if (str.empty()) return 0.0;
    char* end = nullptr;
    errno = 0;
    double p = std::strtod(str.c_str(), &end);
    if (end != str.c_str() + str.size() || errno == ERANGE || !std::isfinite(p)) {
        return 0.0;
    }
    return p;
}

std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return str;
}

} // namespace

std::string time_text(const Row& row) {
    if (row.picture_start_s.empty() && row.picture_end_s.empty()) {
        return "\xe2\x80\x94"; // U+2014 em dash
    }
    return row.picture_start_s + "-" + row.picture_end_s + " s";
}

std::string laya_text(const Row& row) {
    if (row.laya_choice.empty()) {
        return "\xe2\x80\x94"; // U+2014 em dash
    }
    double p = parse_prob(row.laya_probability);
    std::string p_str = format_2f(p);
    if (p_str.size() >= 2 && p_str[0] == '0' && p_str[1] == '.') {
        p_str = p_str.substr(1); // strip leading 0 -> e.g. .91
    }
    return row.laya_choice + " " + p_str;
}

std::string why_text(const Row& row) {
    std::string blue = (row.sound_label == "call")
        ? "bat is higher than baf, so Blue A says call."
        : "bat is not higher than baf, so Blue A says not-call.";

    if (row.mix_reason == "manual") {
        return blue + " The label was set by hand to " + row.mix_label + ".";
    }

    double p = parse_prob(row.laya_probability);
    std::string p_str = format_2f(p);

    if (row.mix_reason == "agree") {
        return blue + " Laya picked " + row.laya_choice + " at " + p_str + ", the same pile, so the label stays.";
    }
    if (row.mix_reason == "disagree") {
        return blue + " Laya picked " + row.laya_choice + " at " + p_str + ", the other pile, so the row goes to review.";
    }
    if (!row.laya_choice.empty()) {
        return blue + " Laya picked " + row.laya_choice + " but only at " + p_str + ", under 0.6, so Blue A's label stands.";
    }
    return blue + " There is no saved Laya choice, so the label stays with Blue A.";
}

std::string search_haystack(const Row& row) {
    std::string combined = row.candidate_id + " " +
                           row.sample_id + " " +
                           time_text(row) + " " +
                           row.laya_choice + " " +
                           row.mix_reason + " " +
                           row.mix_label;
    return to_lower(std::move(combined));
}

} // namespace whalemix::format
