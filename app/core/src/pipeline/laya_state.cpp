#include "whalemix/pipeline/laya_state.h"
#include <cstdio>
#include <vector>

namespace whalemix::pipeline {

namespace {

std::string format_3f(double val) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", val);
    return std::string(buf);
}

} // namespace

std::string state_for_row(const Row& row) {
    return "sample " + row.sample_id + ". " +
           "picture " + row.picture_start_s + " to " + row.picture_end_s + " seconds. " +
           "bat " + row.bat + ". " +
           "baf " + row.baf + ". " +
           "Blue A label " + row.sound_label + ".";
}

std::string state_for_summary(const Summary& summary) {
    std::vector<std::string> lines;
    lines.push_back(std::to_string(summary.rows) + " pictures.");
    lines.push_back(
        "call " + std::to_string(summary.call) + ". " +
        "not-call " + std::to_string(summary.not_call) + ". " +
        "review " + std::to_string(summary.review) + "."
    );
    lines.push_back(
        "agreed " + std::to_string(summary.agree) + ". " +
        "blue-a kept " + std::to_string(summary.blue_a) + ". " +
        "disagreed " + std::to_string(summary.disagree) + "."
    );

    if (summary.highest.has_value()) {
        const auto& h = summary.highest.value();
        lines.push_back(
            "highest bat " + format_3f(h.bat) + " on " + h.sample_id +
            " from " + h.picture_start_s + " to " + h.picture_end_s + " seconds."
        );
    }

    for (const auto& sample : summary.samples) {
        std::string bat_text = sample.highest_bat.has_value()
            ? format_3f(sample.highest_bat.value())
            : "none";
        std::string noise_text;
        if (sample.noise_db.has_value()) {
            noise_text = ", 63 Hz " + sample.noise_db.value();
        }
        lines.push_back(
            sample.sample_id + ": " + std::to_string(sample.pictures) +
            " pictures, highest bat " + bat_text + ", call rows " +
            std::to_string(sample.call) + noise_text + "."
        );
    }

    if (summary.noise_attached) {
        lines.push_back("63 Hz levels are attached by sample.");
    } else {
        lines.push_back("63 Hz levels were not attached.");
    }

    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) out += " ";
        out += lines[i];
    }
    return out;
}

std::string row_question_instructions() {
    return "A 25-second ocean-sound picture has two scores. "
           "bat is the call score. baf is the not-a-call score. "
           "Which pile should this picture go in?";
}

std::string row_question_criteria_call() {
    return "bat is greater than baf";
}

std::string row_question_criteria_not_call() {
    return "baf is greater than or equal to bat";
}

std::string pattern_question_instructions() {
    return "Using only the counts in the state, how does the call score sit across the pictures?";
}

std::string pattern_question_criteria_flat() {
    return "the call scores stay in the same low band across samples";
}

std::string pattern_question_criteria_one_spike() {
    return "one picture is near one half and the rest stay low";
}

std::string pattern_question_criteria_too_few() {
    return "there are not two call pictures in two different noise levels";
}

std::string rate_ready_question_instructions() {
    return "Do these counts support a miss rate, a false-alarm rate, or a noise comparison?";
}

std::string rate_ready_question_criteria_false() {
    return "the table has fewer than two trusted call pictures in each noise level";
}

std::string rate_ready_question_criteria_true() {
    return "the table has at least two trusted call pictures in a quieter level and a noisier level";
}

} // namespace whalemix::pipeline
