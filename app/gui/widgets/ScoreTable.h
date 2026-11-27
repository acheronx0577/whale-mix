#pragma once

#include <QAbstractScrollArea>
#include <QScrollBar>
#include <QSet>
#include <QString>
#include <vector>
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

    // Unified WhaleMix light model (engine runs): grouped WAV rows + 10s windows.
    // Legacy Blue A + Laya CSVs keep the original seven columns. Detected from the first row.
    void set_unified_mode(bool unified);
    void set_threshold(double t) { threshold_ = t; viewport()->update(); }
    bool unified_mode() const { return unified_; }
    int column_count() const { return unified_ ? 6 : 7; }
    QString column_title(int col) const;
    // Display text of a window row's REASON cell (plain English in unified mode).
    QString reason_display(size_t model_index) const;
    int column_left(int col) const { return column_x(col); }
    int column_span(int col) const { return column_width(col); }
    // Stable right gutter so the overlay scrollbar never clips LABEL pills.
    int right_gutter() const;
    int label_pad() const { return kCellPad; }

    void set_group_expanded(int group, bool expanded);
    bool group_expanded(int group) const;
    int group_count() const { return static_cast<int>(groups_.size()); }
    int view_item_count() const { return unified_ ? static_cast<int>(view_items_.size())
                                                  : static_cast<int>(visible_indices_.size()); }

    QScrollBar* overlay_scroll_bar() const { return overlay_bar_; }

    const std::vector<int>& visible_indices() const { return visible_indices_; }
    int visible_count() const { return view_item_count(); }

signals:
    void selection_changed(int model_index);
    // A window row (or a timeline tick) was clicked with the mouse (opens the inspector).
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
    void unified_column_widths(int out[6]) const;

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
    int selected_view_ = -1;

    struct WavGroup {
        QString key;
        QString filename;
        int first = 0;
        int count = 0;
        int call_count = 0;
        int borderline_count = 0;
        double peak = 0.0;
        double duration_s = 0.0;
        bool expanded = false;
        QString label;
        QString summary;
    };
    struct ViewItem {
        enum Type { Group, Window } type = Window;
        int group = -1;
        int model_index = -1;
    };
    std::vector<WavGroup> groups_;
    std::vector<ViewItem> view_items_;
    QSet<QString> expanded_keys_;

    void rebuild_groups_from(size_t from_row);
    void rebuild_all_groups();
    void refresh_group(int g);
    void rebuild_view_items();
    QString group_key(const Row& r) const;
    QString group_filename(const Row& r) const;
    int group_for_model(int model_index) const;
    int view_index_for_model(int model_index) const;
    int tick_at(int group, int x) const;
    QRect timeline_rect(int row_y) const;
    void paint_header(QPainter& p, int w) const;
    void paint_timeline(QPainter& p, const QRect& r, int group, int active_model) const;
    void paint_reason(QPainter& p, const QRect& r, const QString& text, const QColor& color) const;
    void move_selection(int delta_views, int page);

    static constexpr int kHeaderHeight = 34;
    static constexpr int kRowHeight = 52;
    static constexpr int kColWidths[6] = {136, 146, 84, 84, 116, 88}; // col 6 = rest
    static constexpr int kCellPad = 16;  // left padding inside every cell, and minimum right padding
    static constexpr int kLabelGutter = 22; // scrollbar-gutter: stable + right padding
};

} // namespace whalemix::gui
