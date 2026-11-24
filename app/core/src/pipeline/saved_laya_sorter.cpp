#include "whalemix/pipeline/saved_laya_sorter.h"
#include "whalemix/table.h"

namespace whalemix::pipeline {

void SavedLayaSorter::sort(std::vector<Row>& rows, double /*min_probability*/) {
    for (auto& row : rows) {
        apply_saved_laya(row);
    }
}

} // namespace whalemix::pipeline
