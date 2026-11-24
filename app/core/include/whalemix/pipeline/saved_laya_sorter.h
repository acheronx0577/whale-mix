#pragma once

#include "whalemix/pipeline/sorter.h"

namespace whalemix::pipeline {

class SavedLayaSorter : public Sorter {
public:
    std::string name() const override { return "saved-columns"; }
    void sort(std::vector<Row>& rows, double min_probability = rules::kLayaMinProbability) override;
};

} // namespace whalemix::pipeline
