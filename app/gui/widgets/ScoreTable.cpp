#include "ScoreTable.h"
#include "Theme.h"
#include "Fonts.h"
#include "Chip.h"
#include "whalemix/format.h"
#include "DisplayText.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QScrollBar>
#include <QFontMetrics>
#include <QFileInfo>
#include <algorithm>
#include <cmath>

namespace whalemix::gui {

ScoreTable::ScoreTable(QWidget* parent) : QAbstractScrollArea(parent) {
    viewport()->setMouseTracking(true);
    setFrameShape(QFrame::NoFrame);

    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    overlay_bar_ = new QScrollBar(Qt::Vertical, viewport());
    overlay_bar_->setObjectName("scoreTableScrollBar");
    overlay_bar_->setCursor(Qt::ArrowCursor);
    overlay_bar_->setStyleSheet(
        "QScrollBar#scoreTableScrollBar:vertical {"
        "  width: 12px; background: transparent; border: none; margin: 0;"
        "}"
        "QScrollBar#scoreTableScrollBar::handle:vertical {"
        "  background: #CBD5E1; min-height: 32px; border-radius: 3px; margin: 0 3px;"
        "}"
        "QScrollBar#scoreTableScrollBar::handle:vertical:hover,"
        "QScrollBar#scoreTableScrollBar::handle:vertical:pressed {"
        "  background: #94A3B8;"
        "}"
        "QScrollBar#scoreTableScrollBar::add-line:vertical, QScrollBar#scoreTableScrollBar::sub-line:vertical {"
        "  height: 0px; border: none; background: transparent;"
        "}"
        "QScrollBar#scoreTableScrollBar::add-page:vertical, QScrollBar#scoreTableScrollBar::sub-page:vertical {"
        "  background: transparent;"
        "}"
    );
    overlay_bar_->hide();
    connect(verticalScrollBar(), &QScrollBar::rangeChanged, this, [this](int lo, int hi) {
        overlay_bar_->setRange(lo, hi);
        overlay_bar_->setVisible(hi > lo);
    });
    connect(verticalScrollBar(), &QScrollBar::valueChanged, overlay_bar_, &QScrollBar::setValue);
    connect(overlay_bar_, &QScrollBar::valueChanged, verticalScrollBar(), &QScrollBar::setValue);
    verticalScrollBar()->setSingleStep(kRowHeight);
    horizontalScrollBar()->hide();
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

ScoreTable::CachedRow ScoreTable::make_cached_row(const Row& r) {
    CachedRow c;
    std::string raw_name = r.candidate_id.empty() ? r.sample_id : r.candidate_id;
    c.win_name = QString::fromStdString(raw_name);
    c.time_str = QString::fromStdString(format::time_text(r));
    c.bat = QString::fromStdString(r.bat);
    c.baf = QString::fromStdString(r.baf);
    c.laya_str = QString::fromStdString(format::laya_text(r));
    c.reason_str = QString::fromStdString(r.mix_reason);
    c.conf_str = display::confidence_text(r);
    c.label_str = QString::fromStdString(r.mix_label);
    c.chip_size = Chip::size_for_label(c.label_str);
    c.search_haystack_lower = format::search_haystack(r);
    return c;
}

void ScoreTable::cache_row(const Row& r) { cached_rows_.push_back(make_cached_row(r)); }

void ScoreTable::detect_mode() {
    if (!all_rows_.empty()) unified_ = display::is_unified_row(all_rows_.front());
}

void ScoreTable::set_unified_mode(bool unified) {
    if (unified_ == unified) return;
    unified_ = unified;
    if (unified_) rebuild_all_groups();
    update_elided_cache();
    update_filtering();
    viewport()->update();
}

QString ScoreTable::column_title(int col) const {
    static const char* legacy[7] = {"WINDOW", "TIME", "BAT", "BAF", "LAYA", "REASON", "LABEL"};
    static const char* unified[6] = {"WAV FILE / WINDOW", "TIME", "TIMELINE", "PEAK", "REASON", "LABEL"};
    if (col < 0 || col >= column_count()) return QString();
    return QString::fromLatin1(unified_ ? unified[col] : legacy[col]);
}

QString ScoreTable::reason_display(size_t model_index) const {
    if (model_index >= all_rows_.size()) return QString();
    if (!unified_) {
        if (model_index >= cached_rows_.size()) return QString();
        return cached_rows_[model_index].reason_str;
    }
    return display::signal_reason(all_rows_[model_index], threshold_);
}

QString ScoreTable::group_key(const Row& r) const {
    if (!r.audio_path.empty()) return QString::fromStdString(r.audio_path);
    return QString::fromStdString(r.sample_id);
}

QString ScoreTable::group_filename(const Row& r) const {
    if (!r.audio_path.empty()) return QFileInfo(QString::fromStdString(r.audio_path)).fileName();
    const QString id = QString::fromStdString(r.sample_id);
    const int slash = std::max(id.lastIndexOf('/'), id.lastIndexOf('\\'));
    return slash >= 0 ? id.mid(slash + 1) : id;
}

void ScoreTable::refresh_group(int g) {
    if (g < 0 || g >= static_cast<int>(groups_.size())) return;
    auto& gr = groups_[g];
    gr.call_count = 0;
    gr.borderline_count = 0;
    gr.peak = 0.0;
    gr.duration_s = 0.0;
    for (int i = 0; i < gr.count; ++i) {
        const Row& r = all_rows_[gr.first + i];
        if (r.mix_label == "call") gr.call_count++;
        double v = 0.0;
        if (display::parse_score(r.bat, &v)) {
            if (v > gr.peak) gr.peak = v;
            if (v >= display::kBorderlineMin && v < threshold_ && r.mix_label != "call") {
                gr.borderline_count++;
            }
        }
        gr.duration_s = std::max(gr.duration_s, display::parse_seconds(r.picture_end_s));
    }
    const int non = gr.count - gr.call_count;
    gr.label = gr.call_count > 0 ? "call" : "not-call";
    if (gr.call_count > 0) {
        gr.summary = display::summary_reason(gr.call_count, non);
    } else if (gr.count > 0) {
        // Most informative window: peak score's reason.
        int best = gr.first;
        double best_v = -1.0;
        for (int i = 0; i < gr.count; ++i) {
            double v = 0.0;
            if (display::parse_score(all_rows_[gr.first + i].bat, &v) && v > best_v) {
                best_v = v;
                best = gr.first + i;
            }
        }
        gr.summary = display::signal_reason(all_rows_[best], threshold_);
    } else {
        gr.summary.clear();
    }
}

void ScoreTable::rebuild_all_groups() {
    groups_.clear();
    if (!unified_ || all_rows_.empty()) return;
    groups_.reserve(all_rows_.size() / 16 + 1);
    for (size_t i = 0; i < all_rows_.size(); ++i) {
        const QString key = group_key(all_rows_[i]);
        if (groups_.empty() || groups_.back().key != key) {
            WavGroup g;
            g.key = key;
            g.filename = group_filename(all_rows_[i]);
            g.first = static_cast<int>(i);
            g.count = 1;
            g.expanded = expanded_keys_.contains(key);
            groups_.push_back(std::move(g));
        } else {
            groups_.back().count++;
        }
    }
    for (int g = 0; g < static_cast<int>(groups_.size()); ++g) refresh_group(g);
}

void ScoreTable::rebuild_groups_from(size_t from_row) {
    if (!unified_) return;
    if (from_row == 0 || groups_.empty()) {
        rebuild_all_groups();
        return;
    }
    // Extend the last group or start new ones for the newly appended rows.
    for (size_t i = from_row; i < all_rows_.size(); ++i) {
        const QString key = group_key(all_rows_[i]);
        if (groups_.empty() || groups_.back().key != key) {
            if (!groups_.empty()) refresh_group(static_cast<int>(groups_.size()) - 1);
            WavGroup g;
            g.key = key;
            g.filename = group_filename(all_rows_[i]);
            g.first = static_cast<int>(i);
            g.count = 1;
            g.expanded = expanded_keys_.contains(key);
            groups_.push_back(std::move(g));
        } else {
            groups_.back().count++;
        }
    }
    if (!groups_.empty()) refresh_group(static_cast<int>(groups_.size()) - 1);
}

void ScoreTable::rebuild_view_items() {
    view_items_.clear();
    visible_indices_.clear();
    if (!unified_) return;
    const std::string q_std = query_filter_.toStdString();
    const bool filter_pile = (pile_filter_ != "all");
    const bool filter_query = !q_std.empty();
    view_items_.reserve(groups_.size());
    for (int g = 0; g < static_cast<int>(groups_.size()); ++g) {
        const auto& gr = groups_[g];
        if (filter_pile) {
            if (pile_filter_ == "call" && gr.call_count == 0) continue;
            if (pile_filter_ == "not-call" && gr.call_count > 0) continue;
            if (pile_filter_ == "review") continue;
        }
        if (filter_query) {
            bool hit = gr.filename.toLower().toStdString().find(q_std) != std::string::npos;
            if (!hit) {
                for (int i = 0; i < gr.count && !hit; ++i) {
                    const size_t idx = static_cast<size_t>(gr.first + i);
                    if (idx < cached_rows_.size() &&
                        cached_rows_[idx].search_haystack_lower.find(q_std) != std::string::npos) {
                        hit = true;
                    }
                }
            }
            if (!hit) continue;
        }
        ViewItem parent;
        parent.type = ViewItem::Group;
        parent.group = g;
        view_items_.push_back(parent);
        if (gr.expanded) {
            for (int i = 0; i < gr.count; ++i) {
                ViewItem child;
                child.type = ViewItem::Window;
                child.group = g;
                child.model_index = gr.first + i;
                view_items_.push_back(child);
                visible_indices_.push_back(gr.first + i);
            }
        }
    }
}

void ScoreTable::set_rows(const std::vector<Row>& rows, bool has_file) {
    all_rows_ = rows;
    has_file_ = has_file;
    detect_mode();

    cached_rows_.clear();
    cached_rows_.reserve(all_rows_.size());
    for (const auto& r : all_rows_) cache_row(r);

    if (unified_) rebuild_all_groups();
    update_elided_cache();
    update_filtering();
}

void ScoreTable::append_rows(const std::vector<Row>& rows, size_t from, bool has_file) {
    has_file_ = has_file;
    if (from != all_rows_.size() || from > rows.size()) {
        set_rows(rows, has_file);
        return;
    }
    const size_t first_new = all_rows_.size();
    if (first_new == 0 && from < rows.size()) {
        const bool was = unified_;
        unified_ = display::is_unified_row(rows[from]);
        if (was != unified_) viewport()->update();
    }
    all_rows_.insert(all_rows_.end(), rows.begin() + static_cast<std::ptrdiff_t>(from), rows.end());
    for (size_t i = first_new; i < all_rows_.size(); ++i) cache_row(all_rows_[i]);
    update_elided_cache(first_new);

    if (unified_) {
        rebuild_groups_from(first_new);
        update_filtering();
        return;
    }

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

void ScoreTable::update_row(size_t idx, const Row& row) {
    if (idx >= all_rows_.size() || idx >= cached_rows_.size()) return;
    const bool was_label = all_rows_[idx].mix_label != row.mix_label;
    all_rows_[idx] = row;
    CachedRow c = make_cached_row(row);
    QFontMetrics fm_text(fonts::mono(13, 450));
    c.win_name_elided = fm_text.elidedText(c.win_name, Qt::ElideRight, qMax(10, column_width(0) - 20));
    c.time_str_elided = fm_text.elidedText(c.time_str, Qt::ElideRight, qMax(10, column_width(1) - 20));
    cached_rows_[idx] = std::move(c);
    if (unified_) {
        const int g = group_for_model(static_cast<int>(idx));
        if (g >= 0) refresh_group(g);
    }
    if (was_label && pile_filter_ != "all") {
        update_filtering();
    } else {
        viewport()->update();
    }
}

void ScoreTable::update_elided_cache() { update_elided_cache(0); }

void ScoreTable::update_elided_cache(size_t from) {
    if (cached_rows_.size() <= from || unified_) return;  // unified elides visible rows at paint time
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

void ScoreTable::update_filtering() {
    if (unified_) {
        rebuild_view_items();
        update_scroll_bars();
        viewport()->update();
        return;
    }
    visible_indices_.clear();
    std::string q_std = query_filter_.toStdString();
    std::string target_pile = pile_filter_.toStdString();
    bool filter_pile = (pile_filter_ != "all");
    bool filter_query = !q_std.empty();
    visible_indices_.reserve(all_rows_.size());
    for (size_t i = 0; i < all_rows_.size(); ++i) {
        if (filter_pile && all_rows_[i].mix_label != target_pile) continue;
        if (filter_query && i < cached_rows_.size() &&
            cached_rows_[i].search_haystack_lower.find(q_std) == std::string::npos) {
            continue;
        }
        visible_indices_.push_back(static_cast<int>(i));
    }
    update_scroll_bars();
    viewport()->update();
}

int ScoreTable::group_for_model(int model_index) const {
    if (model_index < 0) return -1;
    for (int g = 0; g < static_cast<int>(groups_.size()); ++g) {
        const auto& gr = groups_[g];
        if (model_index >= gr.first && model_index < gr.first + gr.count) return g;
    }
    return -1;
}

int ScoreTable::view_index_for_model(int model_index) const {
    for (int i = 0; i < static_cast<int>(view_items_.size()); ++i) {
        if (view_items_[i].type == ViewItem::Window && view_items_[i].model_index == model_index) return i;
    }
    return -1;
}

void ScoreTable::set_group_expanded(int group, bool expanded) {
    if (group < 0 || group >= static_cast<int>(groups_.size())) return;
    if (groups_[group].expanded == expanded) return;
    groups_[group].expanded = expanded;
    if (expanded) expanded_keys_.insert(groups_[group].key);
    else expanded_keys_.remove(groups_[group].key);
    rebuild_view_items();
    update_scroll_bars();
    viewport()->update();
}

bool ScoreTable::group_expanded(int group) const {
    if (group < 0 || group >= static_cast<int>(groups_.size())) return false;
    return groups_[group].expanded;
}

void ScoreTable::select_model_index(int idx) {
    if (idx >= -1 && idx < static_cast<int>(all_rows_.size())) {
        selected_model_index_ = idx;
        if (unified_ && idx >= 0) {
            const int g = group_for_model(idx);
            if (g >= 0 && !groups_[g].expanded) set_group_expanded(g, true);
            selected_view_ = view_index_for_model(idx);
        }
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
    if (first_call >= 0) select_model_index(first_call);
    else if (!all_rows_.empty()) select_model_index(0);
    else select_model_index(-1);
}

void ScoreTable::move_selection(int delta_views, int page) {
    const int n = view_item_count();
    if (n <= 0) return;
    int pos = -1;
    if (unified_) {
        pos = selected_view_;
        if (pos < 0) pos = view_index_for_model(selected_model_index_);
    } else {
        for (int i = 0; i < static_cast<int>(visible_indices_.size()); ++i) {
            if (visible_indices_[i] == selected_model_index_) { pos = i; break; }
        }
    }
    const int step = (page != 0) ? page : delta_views;
    int target = (pos < 0) ? 0 : qBound(0, pos + step, n - 1);
    if (unified_) {
        selected_view_ = target;
        const auto& it = view_items_[target];
        if (it.type == ViewItem::Window) {
            selected_model_index_ = it.model_index;
            emit selection_changed(selected_model_index_);
        }
        scroll_to_selection();
        viewport()->update();
    } else {
        select_model_index(visible_indices_[target]);
    }
}

void ScoreTable::select_next() { move_selection(1, 0); }
void ScoreTable::select_previous() { move_selection(-1, 0); }
void ScoreTable::page_down() {
    const int rows = qMax(1, (viewport()->height() - kHeaderHeight) / kRowHeight);
    move_selection(0, rows);
}
void ScoreTable::page_up() {
    const int rows = qMax(1, (viewport()->height() - kHeaderHeight) / kRowHeight);
    move_selection(0, -rows);
}
void ScoreTable::select_first() { move_selection(0, -view_item_count()); }
void ScoreTable::select_last() { move_selection(0, view_item_count()); }

void ScoreTable::update_scroll_bars() {
    int content_h = view_item_count() * kRowHeight;
    int view_h = viewport()->height() - kHeaderHeight;
    if (view_h < 0) view_h = 0;
    verticalScrollBar()->setRange(0, qMax(0, content_h - view_h));
    verticalScrollBar()->setPageStep(view_h);
    verticalScrollBar()->setSingleStep(kRowHeight);
    overlay_bar_->setPageStep(view_h);
    overlay_bar_->setSingleStep(kRowHeight);
    place_overlay_bar();
}

void ScoreTable::place_overlay_bar() {
    const int top = kHeaderHeight + kBarInset;
    const int h = viewport()->height() - top - kBarInset;
    overlay_bar_->setGeometry(viewport()->width() - kBarInset - kBarWidth, top, kBarWidth, qMax(0, h));
    overlay_bar_->raise();
}

void ScoreTable::wheelEvent(QWheelEvent* event) {
    if (!event->pixelDelta().isNull()) {
        int dy = -event->pixelDelta().y() * 2;
        verticalScrollBar()->setValue(verticalScrollBar()->value() + dy);
        viewport()->update();
        event->accept();
        return;
    }
    int delta = event->angleDelta().y();
    if (delta != 0) {
        int pixels = -(delta * kRowHeight * 10) / 240;
        verticalScrollBar()->setValue(verticalScrollBar()->value() + pixels);
        viewport()->update();
        event->accept();
        return;
    }
    QAbstractScrollArea::wheelEvent(event);
}

void ScoreTable::scroll_to_selection() {
    int pos = -1;
    if (unified_) {
        pos = selected_view_;
        if (pos < 0) pos = view_index_for_model(selected_model_index_);
    } else if (selected_model_index_ >= 0) {
        for (size_t i = 0; i < visible_indices_.size(); ++i) {
            if (visible_indices_[i] == selected_model_index_) {
                pos = static_cast<int>(i);
                break;
            }
        }
    }
    if (pos < 0) return;
    int row_top = pos * kRowHeight;
    int row_bottom = row_top + kRowHeight;
    int scroll_y = verticalScrollBar()->value();
    int view_h = viewport()->height() - kHeaderHeight;
    if (row_top < scroll_y) verticalScrollBar()->setValue(row_top);
    else if (row_bottom > scroll_y + view_h) verticalScrollBar()->setValue(row_bottom - view_h);
}

int ScoreTable::row_at_pos(int y) const {
    if (y < kHeaderHeight) return -1;
    int scroll_y = verticalScrollBar()->value();
    int adjusted_y = y - kHeaderHeight + scroll_y;
    int idx = adjusted_y / kRowHeight;
    if (idx >= 0 && idx < view_item_count()) return idx;
    return -1;
}

int ScoreTable::right_gutter() const { return unified_ ? kLabelGutter : 0; }

void ScoreTable::unified_column_widths(int out[6]) const {
    const int w = viewport()->width();
    const bool compact = w < 1000;
    int wav = compact ? 168 : 236;
    int time = compact ? 120 : 168;
    int peak = compact ? 72 : 88;
    int label = compact ? 120 : 136;
    const int reason_min = compact ? 140 : 200;
    const int reason_pref = compact ? 168 : 216;
    const int tl_min = compact ? 96 : 170;
    const int gutter = kLabelGutter;

    auto rest_of = [&]() { return w - gutter - wav - time - peak - label; };
    int rest = rest_of();
    int timeline = 0, reason = 0;
    if (rest >= tl_min + reason_pref) {
        reason = reason_pref;
        timeline = rest - reason;
    } else if (rest >= tl_min + reason_min) {
        reason = reason_min;
        timeline = rest - reason;
    } else {
        int deficit = (tl_min + reason_min) - rest;
        const int wav_min = compact ? 132 : 188;
        const int time_min = compact ? 88 : 128;
        const int cut_w = qMin(deficit, qMax(0, wav - wav_min));
        wav -= cut_w;
        deficit -= cut_w;
        const int cut_t = qMin(deficit, qMax(0, time - time_min));
        time -= cut_t;
        rest = rest_of();
        if (rest >= tl_min + reason_min) {
            reason = reason_min;
            timeline = rest - reason;
        } else {
            // Protect REASON (must wrap whole words) and keep a slim TIMELINE.
            const int tl_floor = 56;
            const int reason_floor = 120;
            if (rest >= reason_floor + tl_floor) {
                reason = qMax(reason_floor, rest - tl_min);
                if (reason + tl_floor > rest) reason = rest - tl_floor;
                timeline = rest - reason;
            } else {
                reason = qMax(reason_floor / 2, rest - tl_floor);
                timeline = rest - reason;
            }
        }
    }
    // Never let TIMELINE collide with PEAK: clamp and keep an 8px air gap via padding inside the cell.
    if (timeline < 0) timeline = 0;
    if (reason < 0) reason = 0;
    out[0] = wav;
    out[1] = time;
    out[2] = timeline;
    out[3] = peak;
    out[4] = reason;
    out[5] = label;
}

int ScoreTable::column_x(int col) const {
    int x = 0;
    for (int i = 0; i < col && i < column_count(); ++i) x += column_width(i);
    return x;
}

int ScoreTable::column_width(int col) const {
    if (unified_) {
        if (col < 0 || col > 5) return 0;
        int w[6];
        unified_column_widths(w);
        return w[col];
    }
    if (col >= 0 && col < 6) return kColWidths[col];
    if (col == 6) {
        int total = 0;
        for (int i = 0; i < 6; ++i) total += kColWidths[i];
        return qMax(80, viewport()->width() - total);
    }
    return 0;
}

void ScoreTable::resizeEvent(QResizeEvent* event) {
    QAbstractScrollArea::resizeEvent(event);
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

QRect ScoreTable::timeline_rect(int row_y) const {
    if (!unified_) return QRect();
    const int x = column_x(2) + kCellPad;
    const int w = qMax(8, column_width(2) - 2 * kCellPad);
    return QRect(x, row_y + (kRowHeight - 16) / 2, w, 16);
}

int ScoreTable::tick_at(int group, int x) const {
    if (group < 0 || group >= static_cast<int>(groups_.size())) return -1;
    const int n = groups_[group].count;
    if (n <= 0) return -1;
    const QRect r = timeline_rect(0);  // x is already in cell coords if we pass absolute
    Q_UNUSED(r);
    return -1;  // resolved in mousePress with a real rect
}

void ScoreTable::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    int row = row_at_pos(event->pos().y());
    if (row < 0) return;
    pressed_visible_row_ = row;
    if (unified_) {
        if (row >= static_cast<int>(view_items_.size())) return;
        const auto& it = view_items_[row];
        if (it.type == ViewItem::Group) {
            const int y = kHeaderHeight + row * kRowHeight - verticalScrollBar()->value();
            const QRect tl = timeline_rect(y);
            if (tl.contains(event->pos()) && groups_[it.group].count > 0) {
                const int n = groups_[it.group].count;
                const int inner = qMax(1, tl.width());
                int tick = qBound(0, (event->pos().x() - tl.x()) * n / inner, n - 1);
                set_group_expanded(it.group, true);
                select_model_index(groups_[it.group].first + tick);
                emit row_clicked(selected_model_index_);
            } else {
                selected_view_ = row;
                set_group_expanded(it.group, !groups_[it.group].expanded);
            }
        } else {
            selected_view_ = row;
            select_model_index(it.model_index);
            emit row_clicked(it.model_index);
        }
        viewport()->update();
        return;
    }
    if (row < static_cast<int>(visible_indices_.size())) {
        select_model_index(visible_indices_[row]);
        emit row_clicked(visible_indices_[row]);
        viewport()->update();
    }
}

void ScoreTable::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && pressed_visible_row_ >= 0) {
        pressed_visible_row_ = -1;
        viewport()->update();
    }
}

void ScoreTable::paint_header(QPainter& p, int w) const {
    p.fillRect(0, 0, w, kHeaderHeight, theme::kMist);
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(0, kHeaderHeight, w, kHeaderHeight);
    QFont header_font = fonts::mono(11, 450, 0.08);
    p.setFont(header_font);
    p.setPen(theme::kSlate);
    QFontMetrics fm_hdr(header_font);
    for (int col = 0; col < column_count(); ++col) {
        int cx = column_x(col);
        int ty = (kHeaderHeight - fm_hdr.height()) / 2 + fm_hdr.ascent();
        p.drawText(QRect(cx + kCellPad, 0, qMax(8, column_width(col) - kCellPad), kHeaderHeight),
                   Qt::AlignLeft | Qt::AlignVCenter, column_title(col));
        Q_UNUSED(ty);
    }
}

void ScoreTable::paint_timeline(QPainter& p, const QRect& r, int group, int active_model) const {
    if (group < 0 || group >= static_cast<int>(groups_.size()) || r.width() < 8) return;
    const auto& gr = groups_[group];
    p.setPen(Qt::NoPen);
    p.setBrush(theme::kTimelineTrack);
    p.drawRoundedRect(QRectF(r), 4.0, 4.0);
    if (gr.count <= 0) return;
    const int gap = 1;
    const int inner_h = 12;
    const int ty = r.y() + (r.height() - inner_h) / 2;
    const double tw = std::max(2.0, (r.width() - (gr.count - 1) * gap) / double(gr.count));
    double x = r.x();
    for (int i = 0; i < gr.count; ++i) {
        const Row& row = all_rows_[gr.first + i];
        const auto kind = display::tick_kind(row, threshold_);
        QColor c = theme::kTickMute;
        if (kind == display::TickKind::Call) c = theme::kDeep;
        else if (kind == display::TickKind::Borderline) c = theme::kAmber;
        else if (kind == display::TickKind::Silent) c = theme::kTickSilent;
        const int w = std::max(2, static_cast<int>(std::round(tw)));
        QRect tick(static_cast<int>(std::round(x)), ty, w, inner_h);
        if (tick.right() > r.right()) tick.setRight(r.right());
        p.setBrush(c);
        p.drawRoundedRect(QRectF(tick), 1.0, 1.0);
        if (gr.first + i == active_model) {
            p.setPen(QPen(theme::kInk, 1.0));
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(QRectF(tick).adjusted(0.5, 0.5, -0.5, -0.5), 1.0, 1.0);
            p.setPen(Qt::NoPen);
        }
        x += tw + gap;
        if (x >= r.right()) break;
    }
}

void ScoreTable::paint_reason(QPainter& p, const QRect& r, const QString& text, const QColor& color) const {
    p.setFont(fonts::body(13));
    p.setPen(color);
    p.drawText(r, Qt::TextWordWrap | Qt::AlignLeft | Qt::AlignVCenter, text);
}

void ScoreTable::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(viewport());
    p.setRenderHint(QPainter::Antialiasing, true);
    const int w = viewport()->width();
    const int h = viewport()->height();

    QPainterPath clip_path;
    clip_path.addRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
    p.save();
    p.setClipPath(clip_path);
    p.fillRect(0, 0, w, h, Qt::white);

    const int n_view = view_item_count();
    if (n_view == 0) {
        paint_header(p, w);
        QString empty_msg;
        if (!has_file_) {
            empty_msg = unified_ ? "Scanning\u2026 detections appear here as WAV files finish."
                                 : "Open a Blue A score CSV.";
        } else if (all_rows_.empty()) {
            empty_msg = unified_ ? "No scored 10s windows in this file." : "No scored windows in this file.";
        } else {
            empty_msg = unified_ ? "No WAV files match." : "No windows match.";
        }
        p.setFont(fonts::body(14));
        p.setPen(theme::kSlate);
        p.drawText(QRect(16, kHeaderHeight + 16, w - 32, 40), Qt::AlignLeft | Qt::AlignVCenter, empty_msg);
        p.restore();
        p.setPen(QPen(theme::kLine, 1.0));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
        return;
    }

    const int scroll_y = verticalScrollBar()->value();
    const int start_row = scroll_y / kRowHeight;
    const int end_row = qMin(n_view, (scroll_y + h - kHeaderHeight) / kRowHeight + 2);
    QFont text_font = fonts::mono(13, 450);
    QFontMetrics fm_text(text_font);
    QFont body_font = fonts::body(13);
    QFontMetrics fm_body(body_font);
    QFont name_font = fonts::body(13, 500);
    QFontMetrics fm_name(name_font);

    for (int r = start_row; r < end_row; ++r) {
        const int row_y = kHeaderHeight + r * kRowHeight - scroll_y;
        if (row_y + kRowHeight <= kHeaderHeight || row_y >= h) continue;

        const bool is_hovered = (r == hovered_visible_row_);
        const bool is_pressed = (r == pressed_visible_row_);

        if (unified_) {
            const auto& it = view_items_[r];
            const bool is_group = it.type == ViewItem::Group;
            const bool is_selected = is_group ? (r == selected_view_ && selected_view_ >= 0)
                                              : (it.model_index == selected_model_index_);
            if (is_selected) {
                p.fillRect(0, row_y, w, kRowHeight, is_pressed ? QColor(220, 233, 255) : theme::kMist);
                p.fillRect(0, row_y, 3, kRowHeight, theme::kDeep);
            } else if (is_pressed) {
                p.fillRect(0, row_y, w, kRowHeight, theme::kMist);
            } else if (is_hovered) {
                p.fillRect(0, row_y, w, kRowHeight, theme::kPaper);
            }
            p.setPen(QPen(is_selected ? theme::kDeep : theme::kLine, 1.0));
            p.drawLine(0, row_y + kRowHeight, w - right_gutter(), row_y + kRowHeight);

            if (is_group) {
                const auto& gr = groups_[it.group];
                // Col 0: chevron + filename
                const int cx0 = column_x(0);
                const int chev_x = cx0 + 10;
                p.setPen(QPen(theme::kDeep, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
                const int cy = row_y + kRowHeight / 2;
                QPainterPath chev;
                if (gr.expanded) {
                    chev.moveTo(chev_x, cy - 3);
                    chev.lineTo(chev_x + 6, cy + 3);
                    chev.lineTo(chev_x + 12, cy - 3);
                } else {
                    chev.moveTo(chev_x + 2, cy - 6);
                    chev.lineTo(chev_x + 8, cy);
                    chev.lineTo(chev_x + 2, cy + 6);
                }
                p.drawPath(chev);
                p.setFont(name_font);
                p.setPen(theme::kInk);
                const int name_x = cx0 + 28;
                const int name_w = qMax(10, column_width(0) - 28 - 8);
                p.drawText(QRect(name_x, row_y, name_w, kRowHeight), Qt::AlignLeft | Qt::AlignVCenter,
                           fm_name.elidedText(gr.filename, Qt::ElideMiddle, name_w));

                // Col 1: duration + window count (two lines so nothing wraps mid-word)
                p.setFont(text_font);
                p.setPen(theme::kInk);
                const QString dur = display::duration_text(gr.duration_s);
                const QString wins = gr.count == 1 ? QString("1 window") : QString("%1 windows").arg(gr.count);
                const int cx1 = column_x(1) + kCellPad;
                const int tw = qMax(8, column_width(1) - 2 * kCellPad);
                const int mid = row_y + kRowHeight / 2;
                p.drawText(QRect(cx1, mid - fm_text.height() - 1, tw, fm_text.height()),
                           Qt::AlignLeft | Qt::AlignVCenter, dur);
                p.setPen(theme::kSlate);
                p.drawText(QRect(cx1, mid + 1, tw, fm_text.height()), Qt::AlignLeft | Qt::AlignVCenter, wins);

                // Col 2: timeline — padded so it cannot touch PEAK
                paint_timeline(p, timeline_rect(row_y), it.group, selected_model_index_);

                // Col 3: peak
                p.setFont(text_font);
                p.setPen(theme::kInk);
                p.drawText(QRect(column_x(3) + kCellPad, row_y, qMax(8, column_width(3) - kCellPad), kRowHeight),
                           Qt::AlignLeft | Qt::AlignVCenter, display::percent(gr.peak));

                // Col 4: summary reason
                QColor rc = gr.call_count > 0 ? theme::kDeep
                                              : (gr.borderline_count > 0 ? theme::kAmber : theme::kSlate);
                paint_reason(p, QRect(column_x(4) + kCellPad, row_y + 4,
                                      qMax(8, column_width(4) - 2 * kCellPad), kRowHeight - 8),
                             gr.summary, rc);

                // Col 5: LABEL pill, same left edge as the header
                const int chip_y = row_y + (kRowHeight - 22) / 2;
                const QSize cs = Chip::size_for_label(gr.label);
                Chip::paint(&p, QRect(column_x(5) + kCellPad, chip_y, cs.width(), 22), gr.label);
            } else {
                const int m = it.model_index;
                if (m < 0 || m >= static_cast<int>(all_rows_.size())) continue;
                const Row& row = all_rows_[m];
                const auto& cached = cached_rows_[m];
                const int local = m - groups_[it.group].first + 1;

                // Child rail + #N tag
                p.setPen(Qt::NoPen);
                p.setBrush(theme::kLine);
                p.drawRect(column_x(0) + 18, row_y, 2, kRowHeight);
                const auto kind = display::tick_kind(row, threshold_);
                QColor tag_bg = theme::kMist;
                QColor tag_fg = theme::kInk;
                QColor tag_bd = theme::kLine;
                if (kind == display::TickKind::Call) {
                    tag_bg = theme::kDeep;
                    tag_fg = QColor(Qt::white);
                    tag_bd = theme::kDeep;
                } else if (kind == display::TickKind::Borderline) {
                    tag_bg = QColor("#fff7ed");
                    tag_fg = theme::kAmber;
                    tag_bd = QColor("#fed7aa");
                }
                const QString tag = QString("#%1").arg(local);
                QFont tag_font = fonts::mono(11, 450);
                QFontMetrics fm_tag(tag_font);
                const int tag_w = qMax(28, fm_tag.horizontalAdvance(tag) + 10);
                const QRect tag_r(column_x(0) + 28, row_y + (kRowHeight - 20) / 2, tag_w, 20);
                p.setPen(QPen(tag_bd, 1.0));
                p.setBrush(tag_bg);
                p.drawRoundedRect(QRectF(tag_r), 6.0, 6.0);
                p.setFont(tag_font);
                p.setPen(tag_fg);
                p.drawText(tag_r, Qt::AlignCenter, tag);

                // TIME range
                p.setFont(text_font);
                p.setPen(theme::kInk);
                p.drawText(QRect(column_x(1) + kCellPad, row_y, qMax(8, column_width(1) - 2 * kCellPad), kRowHeight),
                           Qt::AlignLeft | Qt::AlignVCenter, display::window_time_range(row));

                // TIMELINE column left empty on children (keeps PEAK aligned)
                // PEAK / confidence
                p.drawText(QRect(column_x(3) + kCellPad, row_y, qMax(8, column_width(3) - kCellPad), kRowHeight),
                           Qt::AlignLeft | Qt::AlignVCenter, cached.conf_str);

                const QString reason = display::signal_reason(row, threshold_);
                QColor rc = theme::kSlate;
                if (kind == display::TickKind::Call) rc = theme::kDeep;
                else if (kind == display::TickKind::Borderline) rc = theme::kAmber;
                paint_reason(p, QRect(column_x(4) + kCellPad, row_y + 4,
                                      qMax(8, column_width(4) - 2 * kCellPad), kRowHeight - 8),
                             reason, rc);

                const int chip_y = row_y + (kRowHeight - 22) / 2;
                Chip::paint(&p, QRect(column_x(5) + kCellPad, chip_y, cached.chip_size.width(), 22), cached.label_str);
            }
            continue;
        }

        // Legacy flat table
        const int model_idx = visible_indices_[r];
        const bool is_selected = (model_idx == selected_model_index_);
        if (is_selected) {
            p.fillRect(0, row_y, w, kRowHeight, is_pressed ? QColor(220, 233, 255) : theme::kMist);
            p.fillRect(0, row_y, 3, kRowHeight, theme::kDeep);
        } else if (is_pressed) {
            p.fillRect(0, row_y, w, kRowHeight, theme::kMist);
        } else if (is_hovered) {
            p.fillRect(0, row_y, w, kRowHeight, theme::kPaper);
        }
        p.setPen(QPen(is_selected ? theme::kDeep : theme::kLine, 1.0));
        p.drawLine(0, row_y + kRowHeight, w, row_y + kRowHeight);
        if (model_idx < 0 || model_idx >= static_cast<int>(cached_rows_.size())) continue;
        const auto& cached = cached_rows_[model_idx];
        const int ty = row_y + (kRowHeight - fm_text.height()) / 2 + fm_text.ascent();
        p.setFont(text_font);
        p.setPen(theme::kInk);
        p.drawText(column_x(0) + 16, ty, cached.win_name_elided);
        p.drawText(column_x(1) + 16, ty, cached.time_str_elided);
        p.drawText(column_x(2) + 16, ty, cached.bat);
        p.drawText(column_x(3) + 16, ty, cached.baf);
        p.setPen(theme::kSlate);
        p.drawText(column_x(4) + 16, ty, cached.laya_str);
        p.drawText(column_x(5) + 16, ty, fm_text.elidedText(cached.reason_str, Qt::ElideRight,
                                                           qMax(10, column_width(5) - 32)));
        Chip::paint(&p, QRect(column_x(6) + 16, row_y + (kRowHeight - 22) / 2, cached.chip_size.width(), 22),
                    cached.label_str);
    }

    paint_header(p, w);
    p.restore();
    p.setPen(QPen(theme::kLine, 1.0));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 10.0, 10.0);
}

} // namespace whalemix::gui
