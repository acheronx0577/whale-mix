#include "Chip.h"
#include "Theme.h"
#include "Fonts.h"
#include <QFontMetrics>

namespace whalemix::gui {

Chip::Chip(const QString& label, QWidget* parent)
    : QWidget(parent), label_(label) {
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void Chip::set_label(const QString& label) {
    if (label_ != label) {
        label_ = label;
        updateGeometry();
        update();
    }
}

QSize Chip::sizeHint() const {
    return size_for_label(label_);
}

QSize Chip::size_for_label(const QString& label) {
    if (label == "call") {
        static QSize s_call(9 + 5 + 6 + 32 + 10, 22);
        static bool s_init = false;
        if (!s_init) {
            QFont f = fonts::mono(11, 450, 0.1);
            QFontMetrics fm(f);
            s_call = QSize(9 + 5 + 6 + fm.horizontalAdvance("CALL") + 10, 22);
            s_init = true;
        }
        return s_call;
    }
    if (label == "not-call") {
        static QSize s_not_call(9 + 5 + 6 + 56 + 10, 22);
        static bool s_init = false;
        if (!s_init) {
            QFont f = fonts::mono(11, 450, 0.1);
            QFontMetrics fm(f);
            s_not_call = QSize(9 + 5 + 6 + fm.horizontalAdvance("NOT-CALL") + 10, 22);
            s_init = true;
        }
        return s_not_call;
    }
    if (label == "review") {
        static QSize s_review(9 + 5 + 6 + 48 + 10, 22);
        static bool s_init = false;
        if (!s_init) {
            QFont f = fonts::mono(11, 450, 0.1);
            QFontMetrics fm(f);
            s_review = QSize(9 + 5 + 6 + fm.horizontalAdvance("REVIEW") + 10, 22);
            s_init = true;
        }
        return s_review;
    }

    QFont f = fonts::mono(11, 450, 0.1);
    QFontMetrics fm(f);
    int text_w = fm.horizontalAdvance(label.toUpper());
    // 9px left padding + 5px dot + 6px gap + text_w + 10px right padding
    int w = 9 + 5 + 6 + text_w + 10;
    return QSize(w, 22);
}

void Chip::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    paint(&painter, rect(), label_);
}

void Chip::paint(QPainter* p, const QRect& r, const QString& label) {
    if (label.isEmpty()) return;

    p->save();
    p->setRenderHint(QPainter::Antialiasing, true);

    QColor fill;
    QColor border;
    QColor dot_color;
    QColor text_color;

    if (label == "call") {
        fill = theme::kDeep;
        border = theme::kDeep;
        dot_color = theme::kSky;
        text_color = Qt::white;
    } else if (label == "review") {
        fill = theme::kAir;
        border = theme::kAir;
        dot_color = theme::kDeep;
        text_color = theme::kInk;
    } else { // not-call or default
        fill = theme::kMist;
        border = theme::kLine;
        dot_color = theme::kSlate;
        text_color = theme::kInk;
    }

    // Background pill: radius 11
    QRectF pill_rect(r.x() + 0.5, r.y() + 0.5, r.width() - 1.0, 21.0);
    p->setPen(QPen(border, 1.0));
    p->setBrush(fill);
    p->drawRoundedRect(pill_rect, 11.0, 11.0);

    // 5px round dot at x = 9, y centered (22 - 5)/2 = 8.5
    qreal dot_x = r.x() + 9.0;
    qreal dot_y = r.y() + (r.height() - 5.0) / 2.0;
    p->setPen(Qt::NoPen);
    p->setBrush(dot_color);
    p->drawEllipse(QRectF(dot_x, dot_y, 5.0, 5.0));

    // Text: mono 11px DemiBold caps .1em
    QFont f = fonts::mono(11, 450, 0.1);
    p->setFont(f);
    p->setPen(text_color);

    QRect text_rect(r.x() + 9 + 5 + 6, r.y(), r.width() - (9 + 5 + 6 + 10), r.height());
    p->drawText(text_rect, Qt::AlignVCenter | Qt::AlignLeft, label.toUpper());

    p->restore();
}

} // namespace whalemix::gui
