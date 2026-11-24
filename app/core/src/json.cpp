#include "whalemix/json.h"
#include <cstdio>

namespace whalemix::json {

std::string escape_string(const std::string& str) {
    std::string out;
    out.reserve(str.size() + 16);
    for (unsigned char ch : str) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (ch < 0x20) {
                    char hex[8];
                    std::snprintf(hex, sizeof(hex), "\\u%04x", ch);
                    out += hex;
                } else {
                    out += static_cast<char>(ch);
                }
                break;
        }
    }
    return out;
}

} // namespace whalemix::json
