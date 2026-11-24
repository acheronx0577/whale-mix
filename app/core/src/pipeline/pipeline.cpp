#include "whalemix/pipeline/pipeline.h"

namespace whalemix::pipeline {

LoadResult run(Pipeline& pipeline, const ScoreRequest& request) {
    LoadResult result = pipeline.scorer.score(request);
    pipeline.sorter.sort(result.rows);
    return result;
}

} // namespace whalemix::pipeline
