#include "Sidebar.h"
#include "Theme.h"
#include "Fonts.h"
#include "Eyebrow.h"
#include "DisplayText.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QFontMetrics>

namespace whalemix::gui {

Sidebar::Sidebar(QWidget* parent) : QWidget(parent) {
    setFixedWidth(240);
    setMouseTracking(true);
    dither_pixmap_.load(":/dither.png");
}

void Sidebar::set_counts(int call, int not_call, int review, int total) {
    call_count_ = call;
    not_call_count_ = not_call;
    review_count_ = review;
    total_count_ = total;
    update();
}

void Sidebar::set_active_pile(const QString& pile) {
    if (active_pile_ != pile) {
        active_pile_ = pile;
        update();
    }
}

void Sidebar::set_source_info(const QString& file_name, int num_sounds, int num_pictures) {
    file_name_ = file_name;
    num_sounds_ = num_sounds;
    num_pictures_ = num_pictures;
    update();
}

void Sidebar::set_has_manual(bool has_manual) {
    if (has_manual_ != has_manual) {
        has_manual_ = has_manual;
        update();
    }
}

void Sidebar::set_unified_model(bool unified, double threshold) {
    unified_ = unified;
    threshold_ = threshold;
    update();
}

void Sidebar::set_progress(int files_done, int files_total, double eta_s, bool finished) {
    files_done_ = qMax(0, files_done);
    files_total_ = qMax(0, files_total);
    eta_s_ = eta_s;
    finished_ = finished;
    update();
}

QString Sidebar::progress_text() const {
    if (files_total_ <= 0) return QString();
    if (finished_) {
        return display::wav_progress(files_done_, files_total_) + QString(" \u00b7 Done");
    }
    const double pct = 100.0 * files_done_ / files_total_;
    return display::wav_progress(files_done_, files_total_)
        + QString(" \u00b7 %1%").arg(QString::number(pct, 'f', 1));
}

QString Sidebar::source_hint_text() const {
    return display::source_counts(num_sounds_, num_pictures_, unified_);
}

QString Sidebar::eta_display() const {
    if (files_total_ <= 0) return QString();
    if (finished_) return "Scan complete";
    return display::eta_text(eta_s_);
}

void Sidebar::set_compute_mode(const QString& mode) {
    const QString m = mode.toLower();
    if (m != "auto" && m != "hybrid" && m != "gpu" && m != "cpu") return;
    if (compute_mode_ == m) return;
    compute_mode_ = m;
    update();
}

void Sidebar::set_compute_backend(const QString& active, int threads, const QString& note) {
    active_backend_ = active;
    backend_threads_ = threads;
    backend_note_ = note;
    update();
}

QString Sidebar::compute_subtitle() const {
    if (!active_backend_.isEmpty()) {
        QString s = QString("Active: %1").arg(active_backend_.toUpper());
        if (active_backend_.compare("cpu", Qt::CaseInsensitive) == 0 && backend_threads_ > 0) {
            s += QString(" \u00b7 %1 threads").arg(backend_threads_);
        }
        if (compute_mode_ != "cpu" && active_backend_.compare("cpu", Qt::CaseInsensitive) == 0) {
            s += "\nGPU not in this build yet \u00b7 CPU fallback";
        }
        return s;
    }
    if (compute_mode_ == "hybrid") return "CPU + GPU hybrid (falls back to CPU)";
    if (compute_mode_ == "gpu") return "GPU only (falls back to CPU)";
    if (compute_mode_ == "cpu") return "CPU only \u00b7 no GPU needed";
    return "Uses a GPU when found, else CPU";
}

int Sidebar::progress_top() const { return pile_row_rect(3).bottom() + 16; }

int Sidebar::compute_top() const {
    return progress_visible() ? progress_top() + 100 : progress_top();
}

int Sidebar::source_top() const { return compute_top() + 116; }

QRect Sidebar::compute_segment_rect(int index) const {
    if (index < 0 || index > 3) return QRect();
    const int y = compute_top() + 14 + 10;
    return QRect(24 + 2 + index * 47, y + 2, 47, 26);
}

QSize Sidebar::sizeHint() const {
    return QSize(240, 800);
}

QRect Sidebar::pile_row_rect(int index) const {
    // Unified light model: no Review pile, so Total moves up one row.
    int start_y = 60;
    if (unified_ && index == 2) return QRect();
    if (index >= 0 && index <= 2) {
        return QRect(24, start_y + index * 40, 192, 40);
    }
    if (index == 3) {
        return QRect(24, start_y + (unified_ ? 2 : 3) * 40, 192, 42);
    }
    return QRect();
}

void Sidebar::mouseMoveEvent(QMouseEvent* event) {
    int hovered = -1;
    for (int i = 0; i < 3; ++i) { // Only first 3 rows are clickable filters
        if (pile_row_rect(i).contains(event->pos())) {
            hovered = i;
            break;
        }
    }

    int seg = -1;
    for (int i = 0; i < 4; ++i) {
        if (compute_segment_rect(i).contains(event->pos())) seg = i;
    }
    if (hovered != hovered_row_ || seg != hovered_segment_) {
        hovered_row_ = hovered;
        hovered_segment_ = seg;
        setCursor((hovered_row_ >= 0 || seg >= 0) ? Qt::PointingHandCursor : Qt::ArrowCursor);
        update();
    }
}

void Sidebar::leaveEvent(QEvent* /*event*/) {
    if (hovered_row_ >= 0 || pressed_row_ >= 0 || hovered_segment_ >= 0 || pressed_segment_ >= 0) {
        hovered_row_ = -1;
        pressed_row_ = -1;
        hovered_segment_ = -1;
        pressed_segment_ = -1;
        setCursor(Qt::ArrowCursor);
        update();
    }
}

void Sidebar::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        for (int i = 0; i < 3; ++i) {
            if (pile_row_rect(i).contains(event->pos())) {
                pressed_row_ = i;
                update();
                return;
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (compute_segment_rect(i).contains(event->pos())) {
                pressed_segment_ = i;
                update();
                return;
            }
        }
    }
}

void Sidebar::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && pressed_segment_ >= 0) {
        const int seg = pressed_segment_;
        pressed_segment_ = -1;
        if (compute_segment_rect(seg).contains(event->pos())) {
            static const char* modes[4] = {"auto", "hybrid", "gpu", "cpu"};
            const QString m = modes[seg];
            if (m != compute_mode_) {
                compute_mode_ = m;
                active_backend_.clear();  // re-reported by the engine on the next scan
                emit compute_mode_changed(m);
            }
        }
        update();
        return;
    }
    if (event->button() == Qt::LeftButton && pressed_row_ >= 0) {
        int pressed = pressed_row_;
        pressed_row_ = -1;
        update();
        if (pile_row_rect(pressed).contains(event->pos())) {
            QString target_pile;
            if (pressed == 0) target_pile = "call";
            else if (pressed == 1) target_pile = "not-call";
            else if (pressed == 2) target_pile = "review";

            if (active_pile_ == target_pile) {
                active_pile_ = "all";
            } else {
                active_pile_ = target_pile;
            }
            emit pile_filter_changed(active_pile_);
            update();
        }
    }
}

void Sidebar::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    // Background white
    p.fillRect(0, 0, width(), height(), Qt::white);

    // Dither texture at bottom-left behind content
    if (!dither_pixmap_.isNull()) {
        int dither_y = height() - 75;
        p.drawPixmap(0, dither_y, 238, 75, dither_pixmap_);
    }

    // Right divider 1px line
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(width() - 1, 0, width() - 1, height());

    // 1. PILES eyebrow at y = 24
    Eyebrow::paint(&p, QRect(24, 24, 192, 14), "PILES (10S WINDOWS)");

    // 2. Pile bar: y = 46, height = 8, radius = 4, gap = 2
    int bar_y = 46;
    int total = (total_count_ > 0) ? total_count_ : (call_count_ + not_call_count_ + review_count_);
    if (total > 0) {
        int bar_w = 192;
        int active_segments = (call_count_ > 0 ? 1 : 0) + (review_count_ > 0 ? 1 : 0) + (not_call_count_ > 0 ? 1 : 0);
        int available_w = bar_w - (active_segments > 1 ? (active_segments - 1) * 2 : 0);

        int call_w = 0;
        int rev_w = 0;
        int not_w = 0;

        if (call_count_ > 0) {
            call_w = qMax(6, (call_count_ * available_w) / total);
        }
        if (review_count_ > 0) {
            rev_w = qMax(8, (review_count_ * available_w) / total);
        }
        if (not_call_count_ > 0) {
            not_w = available_w - call_w - rev_w;
            if (not_w < 4 && not_call_count_ > 0) not_w = 4;
        }

        // Clip the bar to a rounded rect with radius 4
        QPainterPath clip_path;
        clip_path.addRoundedRect(QRectF(24.0, bar_y, 192.0, 8.0), 4.0, 4.0);
        p.save();
        p.setClipPath(clip_path);

        int cur_x = 24;
        p.setPen(Qt::NoPen);

        // Call (deep)
        if (call_w > 0) {
            p.setBrush(theme::kDeep);
            p.drawRect(cur_x, bar_y, call_w, 8);
            cur_x += call_w + 2;
        }
        // Review (sky)
        if (rev_w > 0) {
            p.setBrush(theme::kSky);
            p.drawRect(cur_x, bar_y, rev_w, 8);
            cur_x += rev_w + 2;
        }
        // Not-call (air)
        if (not_w > 0) {
            p.setBrush(theme::kAir);
            p.drawRect(cur_x, bar_y, not_w, 8);
        }
        p.restore();
    } else {
        // Empty bar in mist
        p.setPen(Qt::NoPen);
        p.setBrush(theme::kMist);
        p.drawRoundedRect(QRectF(24.0, bar_y, 192.0, 8.0), 4.0, 4.0);
    }

    // 3. Pile rows: Call, Not-call, Review, Total
    struct PileRowDef {
        QString name;
        int count;
        QColor dot_color;
        QString pile_id;
    };
    PileRowDef rows[3] = {
        {"Call", call_count_, theme::kDeep, "call"},
        {"Not-call", not_call_count_, theme::kAir, "not-call"},
        {"Review", review_count_, theme::kSky, "review"}
    };

    QFont row_font = fonts::body(14);
    QFont count_font = fonts::mono(14, 450);
    QFontMetrics fm_row(row_font);
    QFontMetrics fm_count(count_font);

    for (int i = 0; i < 3; ++i) {
        QRect r = pile_row_rect(i);
        if (r.isNull()) continue;  // Review pile hidden for the unified model
        bool is_active = (active_pile_ == rows[i].pile_id);
        bool is_hovered = (hovered_row_ == i);
        bool is_pressed = (pressed_row_ == i);

        // Tactile background press tint
        if (is_pressed) {
            p.fillRect(r, QColor(234, 242, 255, 180));
        }

        int y_off = is_pressed ? 1 : 0;

        // Bottom border
        p.setPen(QPen(theme::kLine, 1.0));
        p.drawLine(r.left(), r.bottom(), r.right(), r.bottom());

        // 8px dot (radius 2)
        int dot_y = r.top() + 7 + (fm_row.height() - 8) / 2 + y_off;
        p.setPen(Qt::NoPen);
        p.setBrush(rows[i].dot_color);
        p.drawRoundedRect(QRectF(r.left(), dot_y, 8.0, 8.0), 2.0, 2.0);

        // Name at x = 24 + 8 + 8 = 40
        p.setFont(row_font);
        p.setPen((is_active || is_hovered || is_pressed) ? theme::kDeep : theme::kInk);
        int text_y = r.top() + 7 + fm_row.ascent() + y_off;
        p.drawText(r.left() + 16, text_y, rows[i].name);

        // Count right aligned
        p.setFont(count_font);
        p.setPen(theme::kInk);
        QString cnt_str = display::thousands(rows[i].count);
        int cw = fm_count.horizontalAdvance(cnt_str);
        p.drawText(r.right() - cw, text_y, cnt_str);
    }

    // Total row
    QRect r_tot = pile_row_rect(3);
    QFont tot_font = fonts::heading(14, 450);
    QFont tot_cnt_font = fonts::mono(14, 450);
    QFontMetrics fm_tot(tot_font);
    QFontMetrics fm_tot_cnt(tot_cnt_font);

    p.setFont(tot_font);
    p.setPen(theme::kInk);
    int tot_text_y = r_tot.top() + 4 + fm_tot.ascent();
    p.drawText(r_tot.left(), tot_text_y, "Total windows");

    p.setFont(tot_cnt_font);
    p.setPen(theme::kInk);
    QString tot_str = display::thousands(total);
    int tot_cw = fm_tot_cnt.horizontalAdvance(tot_str);
    p.drawText(r_tot.right() - tot_cw, tot_text_y, tot_str);

    // 4. PROGRESS section (while / after a scan)
    if (progress_visible()) {
        const int py = progress_top();
        Eyebrow::paint(&p, QRect(24, py, 192, 14), finished_ ? "SCAN COMPLETE" : "SCAN PROGRESS");
        // Percent / Done sits on the eyebrow row so "10,000 / 10,000 WAV files" keeps the full 192px column.
        QFont files_font = fonts::body(13);
        QFont pct_font = fonts::mono(12, 450);
        QFontMetrics fm_pct(pct_font);
        const QString pct = finished_ ? QString("Done")
                                      : QString::number(100.0 * files_done_ / files_total_, 'f', 1) + "%";
        const int pct_w = fm_pct.horizontalAdvance(pct);
        p.setFont(pct_font);
        p.setPen(theme::kDeep);
        p.drawText(QRect(216 - pct_w, py, pct_w, 14), Qt::AlignRight | Qt::AlignVCenter, pct);
        p.setFont(files_font);
        p.setPen(theme::kInk);
        const QString files = display::wav_progress(files_done_, files_total_);
        p.drawText(QRect(24, py + 22, 192, 18), Qt::AlignLeft | Qt::AlignVCenter,
                   QFontMetrics(files_font).elidedText(files, Qt::ElideRight, 192));

        const QRectF track(24.0, py + 48, 192.0, 8.0);
        p.setPen(Qt::NoPen);
        p.setBrush(theme::kMist);
        p.drawRoundedRect(track, 4.0, 4.0);
        const double frac = finished_ ? 1.0 : qBound(0.0, static_cast<double>(files_done_) / files_total_, 1.0);
        if (frac > 0.0) {
            p.setBrush(theme::kDeep);
            p.drawRoundedRect(QRectF(track.x(), track.y(), qMax(8.0, track.width() * frac), track.height()), 4.0, 4.0);
        }

        QFont eta_font = fonts::body(13);
        p.setFont(eta_font);
        p.setPen(finished_ ? theme::kDeep : theme::kSlate);
        p.drawText(QRect(24, py + 62, 192, 18), Qt::AlignLeft | Qt::AlignVCenter, eta_display());
    }

    // 5. COMPUTE MODE: Auto / Hybrid / GPU / CPU segmented pill + active backend subtitle
    {
        const int cy = compute_top();
        Eyebrow::paint(&p, QRect(24, cy, 192, 14), "COMPUTE MODE");
        const QRectF outer(24.5, cy + 24.5, 191.0, 29.0);
        p.setPen(QPen(theme::kLine, 1.0));
        p.setBrush(theme::kPaper);
        p.drawRoundedRect(outer, 8.0, 8.0);
        static const char* names[4] = {"Auto", "Hybrid", "GPU", "CPU"};
        static const char* modes[4] = {"auto", "hybrid", "gpu", "cpu"};
        QFont seg_font = fonts::body(12, 500);
        QFontMetrics fm_seg(seg_font);
        p.setFont(seg_font);
        for (int i = 0; i < 4; ++i) {
            const QRect r = compute_segment_rect(i);
            const bool active = compute_mode_ == modes[i];
            if (active) {
                p.setPen(Qt::NoPen);
                p.setBrush(theme::kDeep);
                p.drawRoundedRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5), 6.0, 6.0);
            } else if (pressed_segment_ == i || hovered_segment_ == i) {
                p.setPen(Qt::NoPen);
                p.setBrush(pressed_segment_ == i ? theme::kMist : QColor(Qt::white));
                p.drawRoundedRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5), 6.0, 6.0);
            }
            p.setPen(active ? QColor(Qt::white) : (hovered_segment_ == i ? theme::kDeep : theme::kInk));
            p.drawText(r, Qt::AlignCenter, names[i]);
        }
        QFont sub_font = fonts::body(12);
        p.setFont(sub_font);
        p.setPen(theme::kSlate);
        p.drawText(QRect(24, cy + 60, 192, 36), Qt::TextWordWrap | Qt::AlignLeft | Qt::AlignTop, compute_subtitle());
    }

    // 6. SOURCE section
    int src_eyebrow_y = source_top();
    Eyebrow::paint(&p, QRect(24, src_eyebrow_y, 192, 14), "SOURCE");

    QFont src_name_font = fonts::mono(13, 450);
    p.setFont(src_name_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_src(src_name_font);
    int src_y = src_eyebrow_y + 14 + 7;
    QString disp_file = file_name_.isEmpty() ? "No source yet" : file_name_;
    p.drawText(QRect(24, src_y, 192, 18), Qt::AlignLeft | Qt::AlignVCenter, fm_src.elidedText(disp_file, Qt::ElideRight, 192));

    // Two PILES-style rows so 10,000 / 335,450 never wrap: label left, count right.
    int hint_y = src_y + 18 + 6;
    QFont src_lab = fonts::body(14);
    QFont src_cnt = fonts::mono(14, 450);
    QFontMetrics fm_lab(src_lab);
    QFontMetrics fm_cnt(src_cnt);
    const int row_h = 26;
    auto paint_src_row = [&](int y, const QString& name, const QString& count) {
        p.setFont(src_lab);
        p.setPen(theme::kInk);
        p.drawText(QRect(24, y, 192, row_h), Qt::AlignLeft | Qt::AlignVCenter, name);
        p.setFont(src_cnt);
        const int cw = fm_cnt.horizontalAdvance(count);
        p.drawText(216 - cw, y + (row_h - fm_cnt.height()) / 2 + fm_cnt.ascent(), count);
    };
    paint_src_row(hint_y, "WAV files", display::thousands(num_sounds_));
    const QString right_name = unified_ ? "10s windows" : (num_pictures_ == 1 ? "picture" : "pictures");
    paint_src_row(hint_y + row_h, right_name, display::thousands(num_pictures_));

    // 7. What the labels mean
    int reason_eyebrow_y = hint_y + row_h * 2 + 16;
    if (unified_) {
        Eyebrow::paint(&p, QRect(24, reason_eyebrow_y, 192, 14), "WHAT IT DETECTS");
        int y = reason_eyebrow_y + 14 + 10;
        QFont term_font = fonts::body(13, 600);
        QFont desc_font = fonts::body(13);
        p.setFont(term_font);
        p.setPen(theme::kDeep);
        p.drawText(QRect(24, y, 192, 18), Qt::AlignLeft | Qt::AlignVCenter, "Blue whale A-calls");
        y += 20;
        p.setFont(desc_font);
        p.setPen(theme::kSlate);
        const QString desc = QString("Northeast Pacific population. A 10s window is a Call when "
                                     "WhaleMix Light is at least %1 confident; otherwise it is "
                                     "Below threshold.").arg(display::percent(threshold_));
        QRect bound = p.boundingRect(QRect(24, y, 192, 200), Qt::TextWordWrap | Qt::AlignLeft, desc);
        p.drawText(bound, Qt::TextWordWrap | Qt::AlignLeft, desc);
        y += bound.height() + 6;
        if (has_manual_) {
            const QString man = "Set by hand: you changed it with C or N.";
            QRect mb = p.boundingRect(QRect(24, y, 192, 100), Qt::TextWordWrap | Qt::AlignLeft, man);
            p.drawText(mb, Qt::TextWordWrap | Qt::AlignLeft, man);
        }
        return;
    }

    Eyebrow::paint(&p, QRect(24, reason_eyebrow_y, 192, 14), "REASON");

    struct ReasonDef {
        QString term;
        QString desc;
    };
    std::vector<ReasonDef> reasons = {
        {"blue-a", "No Laya choice, or Laya below 0.6. Blue A\u2019s label stands."},
        {"agree", "Laya picked the same pile as Blue A."},
        {"disagree", "Laya picked the other pile. Row goes to review."}
    };
    if (has_manual_) {
        reasons.push_back({"manual", "Set by hand with C, N or R."});
    }

    QFont term_font = fonts::mono(13, 450);
    QFont desc_font = fonts::body(14);
    int cur_reason_y = reason_eyebrow_y + 14 + 12;

    for (const auto& r : reasons) {
        p.setFont(term_font);
        p.setPen(theme::kDeep);
        p.drawText(QRect(24, cur_reason_y, 192, 20), Qt::AlignLeft | Qt::AlignVCenter, r.term);
        cur_reason_y += 20;

        p.setFont(desc_font);
        p.setPen(theme::kSlate);
        QRect bound = p.boundingRect(QRect(24, cur_reason_y, 192, 200), Qt::TextWordWrap | Qt::AlignLeft, r.desc);
        p.drawText(bound, Qt::TextWordWrap | Qt::AlignLeft, r.desc);
        cur_reason_y += bound.height() + 8;
    }
}

} // namespace whalemix::gui
