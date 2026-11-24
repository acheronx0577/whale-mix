#include "whalemix/csv.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace whalemix::csv {

std::vector<Record> parse_csv(const std::string& text) {
    std::vector<Record> records;
    if (text.empty()) {
        return records;
    }

    size_t i = 0;
    // Check for UTF-8 BOM
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        i = 3;
    }

    std::string field;
    std::vector<std::string> record;
    bool quoted = false;
    size_t line = 1;
    size_t record_line = 1;

    for (; i < text.size(); ++i) {
        char ch = text[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                if (ch == '\n') {
                    ++line;
                }
                field += ch;
            }
            continue;
        }

        if (ch == '"') {
            quoted = true;
        } else if (ch == ',') {
            record.push_back(std::move(field));
            field.clear();
        } else if (ch == '\n' || ch == '\r') {
            if (ch == '\r' && i + 1 < text.size() && text[i + 1] == '\n') {
                ++i;
            }
            record.push_back(std::move(field));
            field.clear();
            // Skip purely blank lines
            if (!(record.size() == 1 && record[0].empty())) {
                records.push_back(Record{std::move(record), record_line});
            }
            record.clear();
            ++line;
            record_line = line;
        } else {
            field += ch;
        }
    }

    if (!field.empty() || !record.empty()) {
        record.push_back(std::move(field));
        if (!(record.size() == 1 && record[0].empty())) {
            records.push_back(Record{std::move(record), record_line});
        }
    }

    return records;
}

std::vector<Record> read_csv_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + path.string() + ": file not found or inaccessible");
    }

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    if (size == 0) {
        throw std::runtime_error(path.string() + " is empty.");
    }
    file.seekg(0, std::ios::beg);

    std::string content(static_cast<size_t>(size), '\0');
    file.read(&content[0], size);

    auto records = parse_csv(content);
    if (records.empty()) {
        throw std::runtime_error(path.string() + " is empty.");
    }

    return records;
}

} // namespace whalemix::csv
