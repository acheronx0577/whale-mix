#include "AppState.h"

namespace whalemix {

AppState::AppState(QObject* parent) : QObject(parent) {}

void AppState::set_rows(std::vector<Row> rows, const QString& source_path) {
    rows_ = std::move(rows);
    source_path_ = source_path;
    has_unsaved_overrides_ = false;

    selected_index_ = -1;
    for (size_t i = 0; i < rows_.size(); ++i) {
        if (rows_[i].mix_label == "call") {
            selected_index_ = static_cast<int>(i);
            break;
        }
    }
    if (selected_index_ == -1 && !rows_.empty()) {
        selected_index_ = 0;
    }

    emit rows_changed();
    emit selection_changed(selected_index_);
}

void AppState::set_selected_index(int idx) {
    if (selected_index_ != idx) {
        selected_index_ = idx;
        emit selection_changed(selected_index_);
    }
}

} // namespace whalemix
