#pragma once

#include "whalemix/pipeline/scorer.h"
#include "whalemix/pipeline/sorter.h"
#include "whalemix/pipeline/reader.h"

namespace whalemix::pipeline {

struct Pipeline {
    Scorer& scorer;
    Sorter& sorter;
    CountReader* reader = nullptr;
};

LoadResult run(Pipeline& pipeline, const ScoreRequest& request);

} // namespace whalemix::pipeline
