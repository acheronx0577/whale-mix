#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace whalemix::csv {

struct Record {
    std::vector<std::string> fields;
    size_t line = 0;
};

std::vector<Record> parse_csv(const std::string& text);
std::vector<Record> read_csv_file(const std::filesystem::path& path);

} // namespace whalemix::csv
