#pragma once

#include <QAbstractScrollArea>
#include <QScrollBar>
#include <vector>
#include <QString>
#include "whalemix/row.h"

namespace whalemix::gui {

class ScoreTable : public QAbstractScrollArea {
    Q_OBJECT
public:
    explicit ScoreTable(QWidget* parent = nullptr);

    void set_rows(const std::vector<Row>& rows, bool has_file);
    // Streaming path: appends rows[from..] without rebuilding the rows already shown, so a long
    // analysis run costs O(new rows) per refresh instead of O(all rows).
    void append_rows(const std::vector<Row>& rows, size_t from, bool has_file);
    size_t row_count() const { return all_rows_.size(); }
    // Replace one row in place (manual label / undo) without rebuilding the other rows.
    void update_row(size_t idx, const Row& row);
    void set_filter(const QString& pile, const QString& query);

    int selected_model_index() const { return selected_model_index_; }
    void select_model_index(int idx);
    void select_first_call_or_first();

    // Keyboard navigation: J/Down and K/Up, PgUp/PgDn, Home/End
    void select_next();
    void select_previous();
    void page_down();
    void page_up();
    void select_first();
    void select_last();

    // Unified WhaleMix light model rows (engine runs): WINDOW, TIME, CONFIDENCE, REASON, LABEL.
    // Legacy Blue A + Laya CSVs keep the original seven columns. Detected from the first row.
    void set_unified_mode(bool unified);
    // Blue A call threshold shown in the plain-English REASON text.
    void set_threshold(double t) { threshold_ = t; viewport()->update(); }
    bool unified_mode() const { return unified_; }
    int column_count() const { return unified_ ? 5 : 7; }
    QString column_title(int col) const;
    // Display text of a row's REASON cell (plain English in unified mode).
    QString reason_display(size_t model_index) const;
    int column_left(int col) const { return column_x(col); }
    int column_span(int col) const { return column_width(col); }

    // Slim scrollbar drawn inside the rounded table card (below the header, inset from the right
    // border). The area's own scrollbar stays hidden and only holds the scroll range and value.
    QScrollBar* overlay_scroll_bar() const { return overlay_bar_; }

    const std::vector<int>& visible_indices() const { return visible_indices_; }
    int visible_count() const { return static_cast<int>(visible_indices_.size()); }

signals:
    void selection_changed(int model_index);
    // A row was clicked with the mouse (opens the inspector).
    void row_clicked(int model_index);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void update_filtering();
    void update_scroll_bars();
    void place_overlay_bar();
    QScrollBar* overlay_bar_ = nullptr;
    static constexpr int kBarWidth = 12;   // transparent track; the handle is 6px
    static constexpr int kBarInset = 5;    // gap to the card's right border and header / bottom
    void scroll_to_selection();
    int row_at_pos(int y) const;
    int column_x(int col) const;
    int column_width(int col) const;

    struct CachedRow {
        QString win_name;
        QString win_name_elided;
        QString time_str;
        QString time_str_elided;
        QString bat;
        QString baf;
        QString laya_str;
        QString reason_str;
        QString conf_str;
        QString label_str;
        QSize chip_size;
        std::string search_haystack_lower;
    };
    static CachedRow make_cached_row(const Row& r);
    void detect_mode();
    bool unified_ = false;
    double threshold_ = 0.897;
    std::vector<CachedRow> cached_rows_;
    void update_elided_cache();
    void update_elided_cache(size_t from);
    void cache_row(const Row& r);

    std::vector<Row> all_rows_;
    std::vector<int> visible_indices_;

    bool has_file_ = false;
    QString pile_filter_ = "all";
    QString query_filter_;

    int selected_model_index_ = -1;
    int hovered_visible_row_ = -1;
    int pressed_visible_row_ = -1;

    static constexpr int kHeaderHeight = 34;
    static constexpr int kRowHeight = 49;
    static constexpr int kColWidths[6] = {136, 146, 84, 84, 116, 88}; // col 6 = rest
    // WINDOW and TIME keep one width (their elisions are cached); CONFIDENCE and LABEL tighten
    // when the inspector is open so REASON keeps room. Col 3 (REASON) = rest.
    static constexpr int kUnifiedWidths[5] = {168, 176, 140, 0, 136};
    static constexpr int kUnifiedCompactWidths[5] = {168, 176, 108, 0, 112};
    static constexpr int kCellPad = 16;  // left padding inside every cell, and minimum right padding
};

} // namespace whalemix::gui
