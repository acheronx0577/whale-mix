#pragma once

#include "whalemix/row.h"
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

namespace whalemix {

struct ExportCounts {
    size_t call = 0;
    size_t not_call = 0;
    size_t review = 0;
    size_t rows = 0;
};

ExportCounts export_rows(
    const std::vector<Row>& rows,
    const std::filesystem::path& out_dir,
    const std::unordered_map<std::string, std::string>& noise = {}
);

std::string decisions_csv(const std::vector<Row>& rows);
std::string decisions_json(const std::vector<Row>& rows);
std::string manifest_text(const std::vector<Row>& rows, const std::string& label);

} // namespace whalemix
