#pragma once

#include "whalemix/row.h"
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>

namespace whalemix {

struct Reading {
    std::string pattern;
    double pattern_probability = 0.0;
    std::string rate_ready;
    double rate_ready_probability = 0.0;
};

struct SampleCount {
    std::string sample_id;
    size_t pictures = 0;
    std::optional<double> highest_bat;
    size_t call = 0;
    std::optional<std::string> noise_db;
};

struct Highest {
    double bat = 0.0;
    std::optional<double> baf;
    std::string sample_id;
    std::string picture_start_s;
    std::string picture_end_s;
};

struct Summary {
    size_t rows = 0;
    size_t call = 0;
    size_t not_call = 0;
    size_t review = 0;
    size_t agree = 0;
    size_t blue_a = 0;
    size_t disagree = 0;
    size_t manual = 0;
    std::vector<SampleCount> samples;
    std::optional<Highest> highest;
    bool noise_attached = false;
};

Summary summarize(
    const std::vector<Row>& rows,
    const std::unordered_map<std::string, std::string>& noise = {}
);

std::string render_report(const Summary& summary, const Reading* reading = nullptr);

} // namespace whalemix
