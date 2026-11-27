#include "Sidebar.h"
#include "Theme.h"
#include "Fonts.h"
#include "Eyebrow.h"
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

QSize Sidebar::sizeHint() const {
    return QSize(240, 800);
}

QRect Sidebar::pile_row_rect(int index) const {
    int start_y = 60;
    if (index >= 0 && index <= 2) {
        return QRect(24, start_y + index * 40, 192, 40);
    }
    if (index == 3) {
        return QRect(24, start_y + 3 * 40, 192, 42);
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

    if (hovered != hovered_row_) {
        hovered_row_ = hovered;
        setCursor((hovered_row_ >= 0) ? Qt::PointingHandCursor : Qt::ArrowCursor);
        update();
    }
}

void Sidebar::leaveEvent(QEvent* /*event*/) {
    if (hovered_row_ >= 0 || pressed_row_ >= 0) {
        hovered_row_ = -1;
        pressed_row_ = -1;
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
    }
}

void Sidebar::mouseReleaseEvent(QMouseEvent* event) {
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
    Eyebrow::paint(&p, QRect(24, 24, 192, 14), "PILES");

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
        QString cnt_str = QString::number(rows[i].count);
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
    p.drawText(r_tot.left(), tot_text_y, "Total");

    p.setFont(tot_cnt_font);
    p.setPen(theme::kInk);
    QString tot_str = QString::number(total);
    int tot_cw = fm_tot_cnt.horizontalAdvance(tot_str);
    p.drawText(r_tot.right() - tot_cw, tot_text_y, tot_str);

    // 4. SOURCE section
    int src_eyebrow_y = r_tot.bottom() + 14;
    Eyebrow::paint(&p, QRect(24, src_eyebrow_y, 192, 14), "SOURCE");

    // Source name
    QFont src_name_font = fonts::mono(13, 450);
    p.setFont(src_name_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_src(src_name_font);
    int src_y = src_eyebrow_y + 14 + 7;
    QString disp_file = file_name_.isEmpty() ? "No source yet" : file_name_;
    p.drawText(QRect(24, src_y, 192, 18), Qt::AlignLeft | Qt::AlignVCenter, fm_src.elidedText(disp_file, Qt::ElideRight, 192));

    // Hint: "<n> sounds · <rows> pictures"
    int hint_y = src_y + 18 + 6;
    QFont hint_font = fonts::body(14);
    p.setFont(hint_font);
    p.setPen(theme::kSlate);
    QString hint_text = QString("%1 sounds · %2 pictures").arg(num_sounds_).arg(num_pictures_);
    p.drawText(QRect(24, hint_y, 192, 18), Qt::AlignLeft | Qt::AlignVCenter, hint_text);

    // 5. REASON section
    int reason_eyebrow_y = hint_y + 18 + 22;
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
