#include "whalemix/pipeline/csv_scorer.h"
#include "whalemix/pipeline/saved_laya_sorter.h"
#include "whalemix/pipeline/pipeline.h"
#include "whalemix/exporter.h"
#include "whalemix/csv.h"
#include <iostream>
#include <string>
#include <unordered_map>

int main(int argc, char* argv[]) {
    std::string scores_path;
    std::string out_path;
    std::string noise_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scores") {
            if (i + 1 < argc) {
                scores_path = argv[++i];
            } else {
                std::cerr << "pass --scores and --out\n";
                return 2;
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                out_path = argv[++i];
            } else {
                std::cerr << "pass --scores and --out\n";
                return 2;
            }
        } else if (arg == "--noise") {
            if (i + 1 < argc) {
                noise_path = argv[++i];
            } else {
                std::cerr << "pass --scores and --out\n";
                return 2;
            }
        } else {
            std::cerr << "unknown argument " << arg << "\n";
            return 2;
        }
    }

    if (scores_path.empty() || out_path.empty()) {
        std::cerr << "pass --scores and --out\n";
        return 2;
    }

    try {
        whalemix::pipeline::CsvScorer scorer;
        whalemix::pipeline::SavedLayaSorter sorter;
        whalemix::pipeline::Pipeline pipeline{scorer, sorter, nullptr};

        whalemix::pipeline::ScoreRequest request{scores_path, {}};
        auto result = whalemix::pipeline::run(pipeline, request);

        std::unordered_map<std::string, std::string> noise_map;
        if (!noise_path.empty()) {
            auto records = whalemix::csv::read_csv_file(noise_path);
            if (!records.empty()) {
                const auto& hdr = records[0].fields;
                size_t id_idx = static_cast<size_t>(-1);
                size_t noise_idx = static_cast<size_t>(-1);
                for (size_t col = 0; col < hdr.size(); ++col) {
                    if (hdr[col] == "sample_id") id_idx = col;
                    else if (hdr[col] == "noise_db" || hdr[col] == "median_db") noise_idx = col;
                }
                for (size_t r = 1; r < records.size(); ++r) {
                    const auto& fields = records[r].fields;
                    if (id_idx < fields.size() && noise_idx < fields.size()) {
                        noise_map[fields[id_idx]] = fields[noise_idx];
                    }
                }
            }
        }

        whalemix::export_rows(result.rows, out_path, noise_map);
        std::cout << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERR: " << e.what() << "\n";
        return 1;
    }
}
