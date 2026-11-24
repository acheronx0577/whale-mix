#include "whalemix/report.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

namespace whalemix {

namespace {

bool parse_double_simple(const std::string& str, double& out) {
    if (str.empty()) return false;
    char* end = nullptr;
    errno = 0;
    double val = std::strtod(str.c_str(), &end);
    if (end != str.c_str() + str.size() || errno == ERANGE || !std::isfinite(val)) {
        return false;
    }
    out = val;
    return true;
}

std::string count_phrase(size_t n, const std::string& word) {
    if (n == 1) {
        return "1 " + word;
    }
    return std::to_string(n) + " " + word + "s";
}

std::string format_3f(double val) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", val);
    return std::string(buf);
}

} // namespace

Summary summarize(
    const std::vector<Row>& rows,
    const std::unordered_map<std::string, std::string>& noise
) {
    std::map<std::string, SampleCount> samples;
    Summary summary;
    summary.rows = rows.size();

    for (const auto& row : rows) {
        if (row.mix_label == "call") summary.call++;
        else if (row.mix_label == "not-call") summary.not_call++;
        else if (row.mix_label == "review") summary.review++;

        if (row.mix_reason == "agree") summary.agree++;
        else if (row.mix_reason == "blue-a") summary.blue_a++;
        else if (row.mix_reason == "disagree") summary.disagree++;
        else if (row.mix_reason == "manual") summary.manual++;

        std::string sample_id = row.sample_id.empty() ? "(no sample)" : row.sample_id;
        auto& slot = samples[sample_id];
        slot.sample_id = sample_id;
        slot.pictures++;
        if (row.mix_label == "call") {
            slot.call++;
        }

        double bat = 0.0;
        if (parse_double_simple(row.bat, bat)) {
            if (!slot.highest_bat.has_value() || bat > slot.highest_bat.value()) {
                slot.highest_bat = bat;
            }
            bool replace = !summary.highest.has_value() || bat > summary.highest->bat;
            if (replace) {
                std::optional<double> baf_val;
                double baf = 0.0;
                if (parse_double_simple(row.baf, baf)) {
                    baf_val = baf;
                }
                summary.highest = Highest{
                    bat,
                    baf_val,
                    slot.sample_id,
                    row.picture_start_s,
                    row.picture_end_s
                };
            }
        }
    }

    for (auto& [id, slot] : samples) {
        auto it = noise.find(id);
        if (it != noise.end()) {
            slot.noise_db = it->second;
            summary.noise_attached = true;
        }
        summary.samples.push_back(std::move(slot));
    }

    return summary;
}

std::string render_report(const Summary& summary, const Reading* reading) {
    std::string opening = (summary.rows == 1)
        ? "This report covers 1 picture."
        : "This report covers " + std::to_string(summary.rows) + " pictures.";

    std::vector<std::string> lines = {
        "# Report",
        "",
        opening,
        "",
        "It uses two tables.",
        "",
        "Blue A's table scores each 25-second picture. `bat` is the call score. `baf` is the not-a-call score. The bigger number is Blue A's answer for that picture.",
        "",
        "Laya's table sorts those written rows. Each row lands in call, not-call, or review.",
        "",
        "## What the tables say",
        ""
    };

    std::vector<std::string> piles;
    if (summary.call > 0) {
        piles.push_back(count_phrase(summary.call, "picture") + " went to call.");
    }
    if (summary.not_call > 0) {
        piles.push_back(count_phrase(summary.not_call, "picture") + " went to not-call.");
    }
    if (summary.review > 0) {
        piles.push_back(count_phrase(summary.review, "picture") + " went to review.");
    }
    if (piles.empty()) {
        piles.push_back("No picture received a label.");
    }

    std::string pile_line;
    for (size_t i = 0; i < piles.size(); ++i) {
        if (i > 0) pile_line += " ";
        pile_line += piles[i];
    }
    lines.push_back(pile_line);
    lines.push_back("");

    if (summary.highest.has_value()) {
        const auto& h = summary.highest.value();
        std::string baf_text = h.baf.has_value() ? format_3f(h.baf.value()) : "missing";
        lines.push_back(
            "The highest call score in Blue A's table is " + format_3f(h.bat) +
            " on " + h.sample_id + ", from " + h.picture_start_s + " to " +
            h.picture_end_s + " seconds. The not-a-call score on that same picture is " +
            baf_text + "."
        );
        lines.push_back("");
    }

    std::vector<std::string> agree;
    if (summary.agree > 0) {
        agree.push_back("On " + count_phrase(summary.agree, "picture") + ", Blue A and Laya gave the same answer.");
    }
    if (summary.blue_a > 0) {
        agree.push_back("On " + count_phrase(summary.blue_a, "picture") + ", Laya was unsure, so Blue A's answer stayed.");
    }
    if (summary.disagree > 0) {
        agree.push_back("On " + count_phrase(summary.disagree, "picture") + ", they disagreed, so the row went to review.");
    }
    if (summary.manual > 0) {
        agree.push_back("On " + count_phrase(summary.manual, "picture") + ", the label was set by hand.");
    }

    if (!agree.empty()) {
        std::string agree_line;
        for (size_t i = 0; i < agree.size(); ++i) {
            if (i > 0) agree_line += " ";
            agree_line += agree[i];
        }
        lines.push_back(agree_line);
        lines.push_back("");
    }

    lines.push_back("## Each sample");
    lines.push_back("");

    for (const auto& sample : summary.samples) {
        std::string bat_text = sample.highest_bat.has_value()
            ? format_3f(sample.highest_bat.value())
            : "none";
        std::string call_text = (sample.call > 0)
            ? count_phrase(sample.call, "picture") + " sorted as call."
            : "None sorted as call.";
        std::string noise_text;
        if (sample.noise_db.has_value()) {
            noise_text = " The 63 Hz level for this sample is " + sample.noise_db.value() + ".";
        }
        lines.push_back(
            "- " + sample.sample_id + ": " + count_phrase(sample.pictures, "picture") +
            ". Highest call score " + bat_text + ". " + call_text + noise_text
        );
    }
    lines.push_back("");

    if (summary.noise_attached) {
        lines.push_back("A 63 Hz level was written next to each sample that had one. That number stands for low-frequency noise.");
    } else {
        lines.push_back("No 63 Hz level was attached to this table.");
    }
    lines.push_back("");

    lines.push_back("These numbers come from the two tables. A miss rate, or a comparison of quieter hours and noisier hours, needs the call shape more than once in a quieter hour and more than once in a noisier hour.");

    if (reading != nullptr) {
        std::string plain;
        if (reading->pattern == "one-spike") {
            plain = " One picture stands out as a call. The rest stay low.";
        } else if (reading->pattern == "flat") {
            plain = " The call scores stay low from picture to picture.";
        } else if (reading->pattern == "too-few") {
            plain = " There are not enough call pictures to compare a quieter hour with a noisier hour.";
        }

        lines.push_back("");
        lines.push_back("## What Laya said about these counts");
        lines.push_back("");
        lines.push_back(
            "Laya's pattern word is " + reading->pattern + "." + plain +
            " The chance Laya gave that word is " + format_3f(reading->pattern_probability) + "."
        );
        lines.push_back("");
        lines.push_back(
            "Can these tables support a miss rate or a noise comparison? Laya said " +
            reading->rate_ready + ". The chance of yes was " +
            format_3f(reading->rate_ready_probability) + "."
        );
        lines.push_back("");
        lines.push_back("Laya read the counts from the two tables. It did not listen to the recordings.");
    }

    lines.push_back("");

    std::string result;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) result += "\n";
        result += lines[i];
    }
    return result;
}

} // namespace whalemix
