#pragma once

#include "whalemix/row.h"
#include "whalemix/rules.h"
#include <string>
#include <vector>

namespace whalemix::pipeline {

class Sorter {
public:
    virtual ~Sorter() = default;
    virtual std::string name() const = 0;
    virtual void sort(std::vector<Row>& rows, double min_probability = rules::kLayaMinProbability) = 0;
};

} // namespace whalemix::pipeline
