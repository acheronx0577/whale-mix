#include "whalemix/pipeline/csv_scorer.h"
#include <stdexcept>

namespace whalemix::pipeline {

LoadResult CsvScorer::score(const ScoreRequest& request) {
    if (!request.images.empty()) {
        throw std::invalid_argument("Images scoring not supported yet in CsvScorer");
    }
    return rows_from_blue_a(request.scores_csv);
}

} // namespace whalemix::pipeline
