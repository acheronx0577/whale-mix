#pragma once

#include <QObject>
#include <QString>
#include <vector>
#include "whalemix/row.h"

namespace whalemix {

class AppState : public QObject {
    Q_OBJECT
public:
    explicit AppState(QObject* parent = nullptr);

    const std::vector<Row>& rows() const { return rows_; }
    std::vector<Row>& mutable_rows() { return rows_; }
    void set_rows(std::vector<Row> rows, const QString& source_path);

    int selected_index() const { return selected_index_; }
    void set_selected_index(int idx);

    QString source_path() const { return source_path_; }
    bool has_unsaved_overrides() const { return has_unsaved_overrides_; }
    void set_unsaved_overrides(bool val) { has_unsaved_overrides_ = val; }

signals:
    void rows_changed();
    void selection_changed(int index);
    void status_message(const QString& msg);

private:
    std::vector<Row> rows_;
    QString source_path_;
    int selected_index_ = -1;
    bool has_unsaved_overrides_ = false;
};

} // namespace whalemix
