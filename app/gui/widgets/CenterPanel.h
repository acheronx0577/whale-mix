#pragma once

#include <QLabel>
#include <QWidget>
#include <QString>
#include <vector>
#include "whalemix/row.h"
#include "ScoreTable.h"
#include "PillButton.h"
#include "SearchBox.h"

namespace whalemix::gui {

class CenterPanel : public QWidget {
    Q_OBJECT
public:
    explicit CenterPanel(QWidget* parent = nullptr);

    ScoreTable* table() { return table_; }
    SearchBox* search_box() { return search_box_; }

    void set_rows(const std::vector<Row>& rows, bool has_file, int num_sounds);
    // Streaming path for engine runs: counts and table cost O(new rows).
    void append_rows(const std::vector<Row>& rows, size_t from, bool has_file, int num_sounds);
    size_t shown_rows() const;
    // One row changed label (manual label / undo): adjust counts and that table row only.
    void update_row(size_t idx, const Row& row, const std::string& old_label);
    void set_active_pile(const QString& pile);
    QString active_pile() const { return active_pile_; }

    void focus_search();

signals:
    void pile_filter_changed(const QString& pile);
    void selection_changed(int model_index);
    void row_clicked(int model_index);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void update_pill_active_states();

    ScoreTable* table_ = nullptr;
    SearchBox* search_box_ = nullptr;

    PillButton* pill_all_ = nullptr;
    PillButton* pill_call_ = nullptr;
    PillButton* pill_not_call_ = nullptr;
    PillButton* pill_review_ = nullptr;

    QString active_pile_ = "all";
    int num_rows_ = 0;
    int num_sounds_ = 0;
    QLabel* meta_label_ = nullptr;
    void update_meta_label();
    int call_count_ = 0;
    int not_call_count_ = 0;
    int review_count_ = 0;
};

} // namespace whalemix::gui
