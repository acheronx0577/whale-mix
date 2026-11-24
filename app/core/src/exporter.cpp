#include "whalemix/exporter.h"
#include "whalemix/report.h"
#include "whalemix/json.h"
#include <fstream>
#include <sstream>

namespace whalemix {

namespace {

std::string csv_quote(const std::string& field) {
    bool need_quote = false;
    for (char c : field) {
        if (c == '"' || c == ',' || c == '\r' || c == '\n') {
            need_quote = true;
            break;
        }
    }
    if (!need_quote) {
        return field;
    }
    std::string q = "\"";
    for (char c : field) {
        if (c == '"') {
            q += "\"\"";
        } else {
            q += c;
        }
    }
    q += '"';
    return q;
}

} // namespace

std::string decisions_csv(const std::vector<Row>& rows) {
    std::string out = "sample_id,candidate_id,picture_start_s,picture_end_s,bat,baf,sound_label,laya_choice,laya_probability,mix_label,mix_reason,image_file\n";
    for (const auto& r : rows) {
        out += csv_quote(r.sample_id) + ",";
        out += csv_quote(r.candidate_id) + ",";
        out += csv_quote(r.picture_start_s) + ",";
        out += csv_quote(r.picture_end_s) + ",";
        out += csv_quote(r.bat) + ",";
        out += csv_quote(r.baf) + ",";
        out += csv_quote(r.sound_label) + ",";
        out += csv_quote(r.laya_choice) + ",";
        out += csv_quote(r.laya_probability) + ",";
        out += csv_quote(r.mix_label) + ",";
        out += csv_quote(r.mix_reason) + ",";
        out += csv_quote(r.image_file) + "\n";
    }
    return out;
}

std::string decisions_json(const std::vector<Row>& rows) {
    if (rows.empty()) {
        return "[]";
    }
    std::string out = "[\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out += "  {\n";
        out += "    \"baf\": \"" + json::escape_string(r.baf) + "\",\n";
        out += "    \"bat\": \"" + json::escape_string(r.bat) + "\",\n";
        out += "    \"candidate_id\": \"" + json::escape_string(r.candidate_id) + "\",\n";
        out += "    \"image_file\": \"" + json::escape_string(r.image_file) + "\",\n";
        out += "    \"laya_choice\": \"" + json::escape_string(r.laya_choice) + "\",\n";
        out += "    \"laya_probability\": \"" + json::escape_string(r.laya_probability) + "\",\n";
        out += "    \"mix_label\": \"" + json::escape_string(r.mix_label) + "\",\n";
        out += "    \"mix_reason\": \"" + json::escape_string(r.mix_reason) + "\",\n";
        out += "    \"picture_end_s\": \"" + json::escape_string(r.picture_end_s) + "\",\n";
        out += "    \"picture_start_s\": \"" + json::escape_string(r.picture_start_s) + "\",\n";
        out += "    \"sample_id\": \"" + json::escape_string(r.sample_id) + "\",\n";
        out += "    \"sound_label\": \"" + json::escape_string(r.sound_label) + "\"\n";
        out += "  }";
        if (i + 1 < rows.size()) {
            out += ",\n";
        } else {
            out += "\n";
        }
    }
    out += "]";
    return out;
}

std::string manifest_text(const std::vector<Row>& rows, const std::string& label) {
    std::string out;
    for (const auto& r : rows) {
        if (r.mix_label != label) {
            continue;
        }
        std::string name;
        if (!r.candidate_id.empty()) {
            name = r.candidate_id;
        } else if (!r.image_file.empty()) {
            name = r.image_file;
        } else {
            name = r.sample_id;
        }
        out += name + "\t" + r.sample_id + "\t" + r.picture_start_s + "\t" + r.picture_end_s + "\n";
    }
    return out;
}

ExportCounts export_rows(
    const std::vector<Row>& rows,
    const std::filesystem::path& out_dir,
    const std::unordered_map<std::string, std::string>& noise
) {
    std::filesystem::create_directories(out_dir);

    // 1. decisions.csv
    {
        std::ofstream f(out_dir / "decisions.csv", std::ios::binary);
        std::string csv_data = decisions_csv(rows);
        f.write(csv_data.data(), csv_data.size());
    }

    // 2. decisions.json
    {
        std::ofstream f(out_dir / "decisions.json", std::ios::binary);
        std::string json_data = decisions_json(rows);
        f.write(json_data.data(), json_data.size());
    }

    // 3. manifests
    ExportCounts counts;
    counts.rows = rows.size();
    for (const std::string& label : {"call", "not-call", "review"}) {
        auto folder = out_dir / label;
        std::filesystem::create_directories(folder);
        std::string text = manifest_text(rows, label);
        if (label == "call") {
            for (const auto& r : rows) if (r.mix_label == "call") counts.call++;
        } else if (label == "not-call") {
            for (const auto& r : rows) if (r.mix_label == "not-call") counts.not_call++;
        } else if (label == "review") {
            for (const auto& r : rows) if (r.mix_label == "review") counts.review++;
        }
        std::ofstream f(folder / "manifest.txt", std::ios::binary);
        if (!text.empty()) {
            f.write(text.data(), text.size());
        }
    }

    // 4. report.md
    {
        Summary sum = summarize(rows, noise);
        std::string rep = render_report(sum, nullptr);
        std::ofstream f(out_dir / "report.md", std::ios::binary);
        f.write(rep.data(), rep.size());
    }

    return counts;
}

} // namespace whalemix
