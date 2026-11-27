#include "ScoreTable.h"
#include "Theme.h"
#include "Fonts.h"
#include "Chip.h"
#include "whalemix/format.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QScrollBar>
#include <QFontMetrics>

namespace whalemix::gui {

ScoreTable::ScoreTable(QWidget* parent) : QAbstractScrollArea(parent) {
    viewport()->setMouseTracking(true);
    setFrameShape(QFrame::NoFrame);

    verticalScrollBar()->setStyleSheet(
        "QScrollBar:vertical {"
        "  width: 10px;"
        "  background: transparent;"
        "  margin: 34px 0 0 0;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #d9e4f7;"
        "  min-height: 24px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical:hover {"
        "  background: #2563eb;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
        "  height: 0px;"
        "}"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
        "  background: transparent;"
        "}"
    );
    verticalScrollBar()->setSingleStep(kRowHeight);
    horizontalScrollBar()->hide();
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void ScoreTable::cache_row(const Row& r) {
    CachedRow c;
    std::string raw_name = r.candidate_id.empty() ? r.sample_id : r.candidate_id;
    c.win_name = QString::fromStdString(raw_name);
    c.time_str = QString::fromStdString(format::time_text(r));
    c.bat = QString::fromStdString(r.bat);
    c.baf = QString::fromStdString(r.baf);
    c.laya_str = QString::fromStdString(format::laya_text(r));
    c.reason_str = QString::fromStdString(r.mix_reason);
    c.label_str = QString::fromStdString(r.mix_label);
    c.chip_size = Chip::size_for_label(c.label_str);
    c.search_haystack_lower = format::search_haystack(r);
    cached_rows_.push_back(std::move(c));
}

void ScoreTable::set_rows(const std::vector<Row>& rows, bool has_file) {
    all_rows_ = rows;
    has_file_ = has_file;

    cached_rows_.clear();
    cached_rows_.reserve(all_rows_.size());
    for (const auto& r : all_rows_) cache_row(r);

    update_elided_cache();
    update_filtering();
}

void ScoreTable::append_rows(const std::vector<Row>& rows, size_t from, bool has_file) {
    has_file_ = has_file;
    if (from != all_rows_.size() || from > rows.size()) {
        set_rows(rows, has_file);  // out of step with the shown rows: rebuild once
        return;
    }
    const size_t first_new = all_rows_.size();
    all_rows_.insert(all_rows_.end(), rows.begin() + static_cast<std::ptrdiff_t>(from), rows.end());
    cached_rows_.reserve(all_rows_.size());
    for (size_t i = first_new; i < all_rows_.size(); ++i) cache_row(all_rows_[i]);
    update_elided_cache(first_new);

    const std::string q_std = query_filter_.toStdString();
    const std::string target_pile = pile_filter_.toStdString();
    const bool filter_pile = (pile_filter_ != "all");
    const bool filter_query = !q_std.empty();
    for (size_t i = first_new; i < all_rows_.size(); ++i) {
        if (filter_pile && all_rows_[i].mix_label != target_pile) continue;
        if (filter_query && cached_rows_[i].search_haystack_lower.find(q_std) == std::string::npos) continue;
        visible_indices_.push_back(static_cast<int>(i));
    }
    update_scroll_bars();
    viewport()->update();
}

void ScoreTable::update_elided_cache() { update_elided_cache(0); }

void ScoreTable::update_elided_cache(size_t from) {
    if (cached_rows_.size() <= from) return;
    QFont text_font = fonts::mono(13, 450);
    QFontMetrics fm_text(text_font);
    int cw0 = qMax(10, column_width(0) - 20);
    int cw1 = qMax(10, column_width(1) - 20);

    for (size_t i = from; i < cached_rows_.size(); ++i) {
        auto& c = cached_rows_[i];
        c.win_name_elided = fm_text.elidedText(c.win_name, Qt::ElideRight, cw0);
        c.time_str_elided = fm_text.elidedText(c.time_str, Qt::ElideRight, cw1);
    }
}

void ScoreTable::set_filter(const QString& pile, const QString& query) {
    pile_filter_ = pile;
    query_filter_ = query.trimmed().toLower();
    update_filtering();
}

void ScoreTable::select_model_index(int idx) {
    if (idx >= -1 && idx < static_cast<int>(all_rows_.size())) {
        selected_model_index_ = idx;
        emit selection_changed(selected_model_index_);
        scroll_to_selection();
        viewport()->update();
    }
}

void ScoreTable::select_first_call_or_first() {
    int first_call = -1;
    for (size_t i = 0; i < all_rows_.size(); ++i) {
        if (all_rows_[i].mix_label == "call") {
            first_call = static_cast<int>(i);
            break;
        }
    }
    if (first_call >= 0) {
        select_model_index(first_call);
    } else if (!all_rows_.empty()) {
        select_model_index(0);
    } else {
        select_model_index(-1);
    }
}

void ScoreTable::select_next() {
    if (visible_indices_.empty()) return;

    int current_pos = -1;
    for (size_t i = 0; i < visible_indices_.size(); ++i) {
        if (visible_indices_[i] == selected_model_index_) {
            current_pos = static_cast<int>(i);
            break;
        }
    }

    if (current_pos < 0) {
        select_model_index(visible_indices_[0]);
    } else if (current_pos + 1 < static_cast<int>(visible_indices_.size())) {
        select_model_index(visible_indices_[current_pos + 1]);
    }
}

void ScoreTable::select_previous() {
    if (visible_indices_.empty()) return;

    int current_pos = -1;
    for (size_t i = 0; i < visible_indices_.size(); ++i) {
        if (visible_indices_[i] == selected_model_index_) {
            current_pos = static_cast<int>(i);
            break;
        }
    }

    if (current_pos < 0) {
        select_model_index(visible_indices_[0]);
    } else if (current_pos - 1 >= 0) {
        select_model_index(visible_indices_[current_pos - 1]);
    }
}

void ScoreTable::page_down() {
    if (visible_indices_.empty()) return;
    int rows_per_page = qMax(1, (viewport()->height() - kHeaderHeight) / kRowHeight);
    int current_pos = -1;
    for (size_t i = 0; i < visible_indices_.size(); ++i) {
        if (visible_indices_[i] == selected_model_index_) {
            current_pos = static_cast<int>(i);
            break;
        }
    }
    int target = (current_pos < 0) ? 0 : qMin(static_cast<int>(visible_indices_.size()) - 1, current_pos + rows_per_page);
    select_model_index(visible_indices_[target]);
}

void ScoreTable::page_up() {
    if (visible_indices_.empty()) return;
    int rows_per_page = qMax(1, (viewport()->height() - kHeaderHeight) / kRowHeight);
    int current_pos = -1;
    for (size_t i = 0; i < visible_indices_.size(); ++i) {
        if (visible_indices_[i] == selected_model_index_) {
            current_pos = static_cast<int>(i);
            break;
        }
    }
    int target = (current_pos < 0) ? 0 : qMax(0, current_pos - rows_per_page);
    select_model_index(visible_indices_[target]);
}

void ScoreTable::select_first() {
    if (!visible_indices_.empty()) {
        select_model_index(visible_indices_.front());
    }
}

void ScoreTable::select_last() {
    if (!visible_indices_.empty()) {
        select_model_index(visible_indices_.back());
    }
}

void ScoreTable::update_filtering() {
    visible_indices_.clear();
    std::string q_std = query_filter_.toStdString();
    std::string target_pile = pile_filter_.toStdString();
    bool filter_pile = (pile_filter_ != "all");
    bool filter_query = !q_std.empty();

    visible_indices_.reserve(all_rows_.size());
    for (size_t i = 0; i < all_rows_.size(); ++i) {
        // Pile filter
        if (filter_pile && all_rows_[i].mix_label != target_pile) {
            continue;
        }

        // Fast zero-allocation query filter using pre-computed search haystack
        if (filter_query && i < cached_rows_.size()) {
            if (cached_rows_[i].search_haystack_lower.find(q_std) == std::string::npos) {
                continue;
            }
        }

        visible_indices_.push_back(static_cast<int>(i));
    }

    update_scroll_bars();
    viewport()->update();
}

void ScoreTable::update_scroll_bars() {
    int content_h = static_cast<int>(visible_indices_.size()) * kRowHeight;
    int view_h = viewport()->height() - kHeaderHeight;
    if (view_h < 0) view_h = 0;

    verticalScrollBar()->setRange(0, qMax(0, content_h - view_h));
    verticalScrollBar()->setPageStep(view_h);
    verticalScrollBar()->setSingleStep(kRowHeight);
}

void ScoreTable::wheelEvent(QWheelEvent* event) {
    if (!event->pixelDelta().isNull()) {
        // High-precision touchpads report small pixel deltas; scale smoothly so swiping isn't painful
        int dy = -event->pixelDelta().y() * 2;
        verticalScrollBar()->setValue(verticalScrollBar()->value() + dy);
        viewport()->update();
        event->accept();
        return;
    }

    int delta = event->angleDelta().y();
    if (delta != 0) {
        // Fast, comfortable row scrolling: 5 rows per standard wheel notch (120 units)
        int pixels = -(delta * kRowHeight * 10) / 240;
        verticalScrollBar()->setValue(verticalScrollBar()->value() + pixels);
        viewport()->update();
        event->accept();
        return;
    }
    QAbstractScrollArea::wheelEvent(event);
}

void ScoreTable::scroll_to_selection() {
    if (selected_model_index_ < 0) return;

    int pos = -1;
    for (size_t i = 0; i < visible_indices_.size(); ++i) {
        if (visible_indices_[i] == selected_model_index_) {
            pos = static_cast<int>(i);
            break;
        }
    }
    if (pos < 0) return; // Selected row is filtered out; keep selection

    int row_top = pos * kRowHeight;
    int row_bottom = row_top + kRowHeight;
    int scroll_y = verticalScrollBar()->value();
    int view_h = viewport()->height() - kHeaderHeight;

    if (row_top < scroll_y) {
        verticalScrollBar()->setValue(row_top);
    } else if (row_bottom > scroll_y + view_h) {
        verticalScrollBar()->setValue(row_bottom - view_h);
    }
}

int ScoreTable::row_at_pos(int y) const {
    if (y < kHeaderHeight) return -1;
    int scroll_y = verticalScrollBar()->value();
    int adjusted_y = y - kHeaderHeight + scroll_y;
    int idx = adjusted_y / kRowHeight;
    if (idx >= 0 && idx < static_cast<int>(visible_indices_.size())) {
        return idx;
    }
    return -1;
}

int ScoreTable::column_x(int col) const {
    int x = 0;
    for (int i = 0; i < col && i < 6; ++i) {
        x += kColWidths[i];
    }
    return x;
}

int ScoreTable::column_width(int col) const {
    if (col >= 0 && col < 6) {
        return kColWidths[col];
    }
    if (col == 6) {
        int total_first_6 = 0;
        for (int i = 0; i < 6; ++i) total_first_6 += kColWidths[i];
        return qMax(80, viewport()->width() - total_first_6);
    }
    return 0;
}

void ScoreTable::resizeEvent(QResizeEvent* event) {
    QAbstractScrollArea::resizeEvent(event);
    update_elided_cache();
    update_scroll_bars();
}

void ScoreTable::mouseMoveEvent(QMouseEvent* event) {
    int row = row_at_pos(event->pos().y());
    if (row != hovered_visible_row_) {
        hovered_visible_row_ = row;
        viewport()->update();
    }
}

void ScoreTable::leaveEvent(QEvent* /*event*/) {
    if (hovered_visible_row_ >= 0 || pressed_visible_row_ >= 0) {
        hovered_visible_row_ = -1;
        pressed_visible_row_ = -1;
        viewport()->update();
    }
}

void ScoreTable::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        int row = row_at_pos(event->pos().y());
        if (row >= 0 && row < static_cast<int>(visible_indices_.size())) {
            pressed_visible_row_ = row;
            select_model_index(visible_indices_[row]);
            viewport()->update();
        }
    }
}

void ScoreTable::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && pressed_visible_row_ >= 0) {
        pressed_visible_row_ = -1;
        viewport()->update();
    }
}

void ScoreTable::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(viewport());
    p.setRenderHint(QPainter::Antialiasing, true);

    int w = viewport()->width();
    int h = viewport()->height();

    // 10px rounded clipped container
    QPainterPath clip_path;
    clip_path.addRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
    p.save();
    p.setClipPath(clip_path);

    // Background white
    p.fillRect(0, 0, w, h, Qt::white);

    // Empty states
    if (visible_indices_.empty()) {
        // Draw header first
        p.fillRect(0, 0, w, kHeaderHeight, theme::kMist);
        p.setPen(QPen(theme::kLine, 1.0));
        p.drawLine(0, kHeaderHeight, w, kHeaderHeight);

        // Header titles
        QFont header_font = fonts::mono(11, 450, 0.12);
        p.setFont(header_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_hdr(header_font);

        const char* titles[7] = {"WINDOW", "TIME", "BAT", "BAF", "LAYA", "REASON", "LABEL"};
        for (int col = 0; col < 7; ++col) {
            int cx = column_x(col);
            int cw = column_width(col);
            int ty = (kHeaderHeight - fm_hdr.height()) / 2 + fm_hdr.ascent();
            p.drawText(cx + 16, ty, titles[col]);
        }

        // Empty message
        QString empty_msg;
        if (!has_file_) {
            empty_msg = "Open a Blue A score CSV.";
        } else if (all_rows_.empty()) {
            empty_msg = "No scored windows in this file.";
        } else {
            empty_msg = "No windows match.";
        }

        QFont empty_font = fonts::body(14);
        p.setFont(empty_font);
        p.setPen(theme::kSlate);
        p.drawText(QRect(16, kHeaderHeight + 16, w - 32, 40), Qt::AlignLeft | Qt::AlignVCenter, empty_msg);

        p.restore();

        // 1px line border around rounded container
        p.setPen(QPen(theme::kLine, 1.0));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
        return;
    }

    // Draw visible rows with scrolling
    int scroll_y = verticalScrollBar()->value();
    int start_row = scroll_y / kRowHeight;
    int end_row = qMin(static_cast<int>(visible_indices_.size()), (scroll_y + h - kHeaderHeight) / kRowHeight + 2);

    QFont text_font = fonts::mono(13, 450);
    QFontMetrics fm_text(text_font);

    for (int r = start_row; r < end_row; ++r) {
        int model_idx = visible_indices_[r];
        const auto& row_data = all_rows_[model_idx];

        int row_y = kHeaderHeight + r * kRowHeight - scroll_y;
        if (row_y + kRowHeight <= kHeaderHeight || row_y >= h) continue;

        bool is_selected = (model_idx == selected_model_index_);
        bool is_hovered = (r == hovered_visible_row_);
        bool is_pressed = (r == pressed_visible_row_);

        // Row background
        if (is_selected) {
            p.fillRect(0, row_y, w, kRowHeight, is_pressed ? QColor(220, 233, 255) : theme::kMist);
            // 3px deep left bar
            p.fillRect(0, row_y, 3, kRowHeight, theme::kDeep);
        } else if (is_pressed) {
            p.fillRect(0, row_y, w, kRowHeight, theme::kMist);
        } else if (is_hovered) {
            p.fillRect(0, row_y, w, kRowHeight, theme::kPaper);
        }

        // Bottom border
        p.setPen(QPen(is_selected ? theme::kDeep : theme::kLine, 1.0));
        p.drawLine(0, row_y + kRowHeight, w, row_y + kRowHeight);

        // Text baseline Y
        int ty = row_y + (kRowHeight - fm_text.height()) / 2 + fm_text.ascent();

        if (model_idx < 0 || model_idx >= static_cast<int>(cached_rows_.size())) continue;
        const auto& cached = cached_rows_[model_idx];

        // Col 0: WINDOW
        {
            int cx = column_x(0);
            p.setFont(text_font);
            p.setPen(theme::kInk);
            p.drawText(cx + 16, ty, cached.win_name_elided);
        }

        // Col 1: TIME
        {
            int cx = column_x(1);
            p.setFont(text_font);
            p.setPen(theme::kInk);
            p.drawText(cx + 16, ty, cached.time_str_elided);
        }

        // Col 2: BAT
        {
            int cx = column_x(2);
            p.setFont(text_font);
            p.setPen(theme::kInk);
            p.drawText(cx + 16, ty, cached.bat);
        }

        // Col 3: BAF
        {
            int cx = column_x(3);
            p.setFont(text_font);
            p.setPen(theme::kInk);
            p.drawText(cx + 16, ty, cached.baf);
        }

        // Col 4: LAYA (slate)
        {
            int cx = column_x(4);
            p.setFont(text_font);
            p.setPen(theme::kSlate);
            p.drawText(cx + 16, ty, cached.laya_str);
        }

        // Col 5: REASON (slate)
        {
            int cx = column_x(5);
            p.setFont(text_font);
            p.setPen(theme::kSlate);
            p.drawText(cx + 16, ty, cached.reason_str);
        }

        // Col 6: LABEL (Chip)
        {
            int cx = column_x(6);
            int chip_y = row_y + (kRowHeight - 22) / 2;
            Chip::paint(&p, QRect(cx + 16, chip_y, cached.chip_size.width(), 22), cached.label_str);
        }
    }

    // Fixed Header pinned on top: 34px high on mist
    p.fillRect(0, 0, w, kHeaderHeight, theme::kMist);
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(0, kHeaderHeight, w, kHeaderHeight);

    QFont header_font = fonts::mono(11, 450, 0.12);
    p.setFont(header_font);
    p.setPen(theme::kSlate);
    QFontMetrics fm_hdr(header_font);

    const char* titles[7] = {"WINDOW", "TIME", "BAT", "BAF", "LAYA", "REASON", "LABEL"};
    for (int col = 0; col < 7; ++col) {
        int cx = column_x(col);
        int ty = (kHeaderHeight - fm_hdr.height()) / 2 + fm_hdr.ascent();
        p.drawText(cx + 16, ty, titles[col]);
    }

    p.restore();

    // 1px line outer border
    p.setPen(QPen(theme::kLine, 1.0));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
}

} // namespace whalemix::gui
