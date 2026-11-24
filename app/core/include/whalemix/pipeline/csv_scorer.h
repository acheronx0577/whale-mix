#pragma once

#include "whalemix/pipeline/scorer.h"

namespace whalemix::pipeline {

class CsvScorer : public Scorer {
public:
    std::string name() const override { return "csv"; }
    LoadResult score(const ScoreRequest& request) override;
};

} // namespace whalemix::pipeline
