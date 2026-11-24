#include "whalemix/table.h"
#include "whalemix/csv.h"
#include "whalemix/rules.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace whalemix {

namespace {

std::string first_present(std::initializer_list<std::string_view> list) {
    for (auto sv : list) {
        if (!sv.empty()) {
            return std::string(sv);
        }
    }
    return "";
}

bool parse_double_strict(std::string_view sv, double& out_score) {
    if (sv.empty()) return false;
    if (sv.front() == ' ' || sv.front() == '\t' || sv.back() == ' ' || sv.back() == '\t') {
        return false;
    }
    std::string s(sv);
    char* end = nullptr;
    errno = 0;
    double val = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || errno == ERANGE || !std::isfinite(val)) {
        return false;
    }
    out_score = val;
    return true;
}

bool parse_score(std::string_view primary, std::string_view alias, double& out_score) {
    std::string_view sv = !primary.empty() ? primary : alias;
    if (sv.empty()) {
        return false;
    }
    return parse_double_strict(sv, out_score);
}

std::string format_6f(double val) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", val);
    return std::string(buf);
}

} // namespace

void apply_saved_laya(Row& row) {
    double probability = 0.0;
    if (!row.laya_probability.empty()) {
        double p = 0.0;
        if (parse_double_strict(row.laya_probability, p)) {
            probability = p;
        }
    }

    auto [label, reason] = rules::mix_label(row.sound_label, row.laya_choice, probability);
    if (!row.laya_choice.empty()) {
        row.laya_probability = format_6f(probability);
    }
    row.mix_label = label;
    row.mix_reason = reason;
    row.auto_label = label;
    row.auto_reason = reason;
}

LoadResult rows_from_blue_a(const std::filesystem::path& path) {
    auto records = csv::read_csv_file(path);
    if (records.empty()) {
        throw std::runtime_error(path.string() + " is empty.");
    }

    const auto& header = records[0].fields;
    std::unordered_map<std::string, size_t> col_indices;
    for (size_t i = 0; i < header.size(); ++i) {
        col_indices[header[i]] = i;
    }

    LoadResult result;
    for (size_t r = 1; r < records.size(); ++r) {
        const auto& rec = records[r];
        size_t prev_count = records[r - 1].fields.size();
        if (rec.fields.size() != prev_count) {
            std::ostringstream oss;
            oss << "CSV error: record " << r << " (line: " << rec.line
                << "): found record with " << rec.fields.size()
                << " fields, but the previous record has " << prev_count << " fields";
            throw std::runtime_error(oss.str());
        }

        auto get_field = [&](const std::string& name) -> std::string_view {
            auto it = col_indices.find(name);
            if (it != col_indices.end() && it->second < rec.fields.size()) {
                return rec.fields[it->second];
            }
            return "";
        };

        std::string_view status = get_field("status");
        if (status.empty()) {
            status = "scored";
        }
        if (status != "scored") {
            result.skipped_not_scored++;
            continue;
        }

        double bat = 0.0;
        if (!parse_score(get_field("bat"), get_field("bat_score"), bat)) {
            result.skipped_bad_score++;
            continue;
        }

        double baf = 0.0;
        if (!parse_score(get_field("baf"), get_field("baf_score"), baf)) {
            result.skipped_bad_score++;
            continue;
        }

        Row row;
        row.sample_id = std::string(get_field("sample_id"));
        row.candidate_id = std::string(get_field("candidate_id"));
        row.picture_start_s = first_present({
            get_field("centered_window_start_s"),
            get_field("picture_start_s"),
            get_field("proposal_start_s")
        });
        row.picture_end_s = first_present({
            get_field("centered_window_end_s"),
            get_field("picture_end_s"),
            get_field("proposal_end_s")
        });
        row.bat = format_6f(bat);
        row.baf = format_6f(baf);
        row.sound_label = rules::sound_label(bat, baf);
        row.image_file = std::string(get_field("image_file"));
        row.laya_choice = std::string(get_field("laya_choice"));
        row.laya_probability = std::string(get_field("laya_probability"));

        apply_saved_laya(row);
        result.rows.push_back(std::move(row));
    }

    return result;
}

void set_label(Row& row, const std::string& label) {
    if (row.mix_label == label) {
        return;
    }
    row.mix_label = label;
    row.mix_reason = "manual";
}

void clear_override(Row& row) {
    row.mix_label = row.auto_label;
    row.mix_reason = row.auto_reason;
}

} // namespace whalemix
