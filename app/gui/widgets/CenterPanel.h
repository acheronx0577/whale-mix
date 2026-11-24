#pragma once

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
    void set_active_pile(const QString& pile);
    QString active_pile() const { return active_pile_; }

    void focus_search();

signals:
    void pile_filter_changed(const QString& pile);
    void selection_changed(int model_index);

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
};

} // namespace whalemix::gui
