#pragma once

#include "whalemix/row.h"
#include <string>

namespace whalemix::format {

std::string time_text(const Row& row);
std::string laya_text(const Row& row);
std::string why_text(const Row& row);
std::string search_haystack(const Row& row);

} // namespace whalemix::format
