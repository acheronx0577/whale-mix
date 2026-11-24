#pragma once

#include "whalemix/row.h"
#include <vector>
#include <string>
#include <filesystem>

namespace whalemix {

struct LoadResult {
    std::vector<Row> rows;
    size_t skipped_not_scored = 0;
    size_t skipped_bad_score = 0;
};

LoadResult rows_from_blue_a(const std::filesystem::path& path);
void apply_saved_laya(Row& row);

void set_label(Row& row, const std::string& label);
void clear_override(Row& row);

} // namespace whalemix
