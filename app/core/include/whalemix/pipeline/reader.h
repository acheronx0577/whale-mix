#pragma once

#include "whalemix/report.h"
#include <optional>

namespace whalemix::pipeline {

class CountReader {
public:
    virtual ~CountReader() = default;
    virtual std::optional<Reading> read(const Summary& summary) = 0;
};

} // namespace whalemix::pipeline
