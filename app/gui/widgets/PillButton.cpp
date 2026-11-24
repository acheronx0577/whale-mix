#include "PillButton.h"
#include "Theme.h"
#include "Fonts.h"
#include <QFontMetrics>
#include <QMouseEvent>

namespace whalemix::gui {

PillButton::PillButton(const QString& name, int count, QWidget* parent)
    : QWidget(parent), name_(name), count_(count) {
    setCursor(Qt::PointingHandCursor);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void PillButton::set_name(const QString& name) {
    if (name_ != name) {
        name_ = name;
        updateGeometry();
        update();
    }
}

void PillButton::set_count(int count) {
    if (count_ != count) {
        count_ = count;
        updateGeometry();
        update();
    }
}

void PillButton::set_active(bool active) {
    if (active_ != active) {
        active_ = active;
        update();
    }
}

QSize PillButton::sizeHint() const {
    QFont fn = fonts::sans(15, 450, -0.02);
    QFontMetrics fmn(fn);
    int nw = fmn.horizontalAdvance(name_);

    QFont fc = fonts::mono(11, 450);
    QFontMetrics fmc(fc);
    int cw = fmc.horizontalAdvance(QString::number(count_));

    // 14px left padding + name + 3px gap + count + 14px right padding
    int w = 14 + nw + 3 + cw + 14;
    return QSize(w, 32);
}

void PillButton::enterEvent(QEnterEvent* /*event*/) {
    hovered_ = true;
    update();
}

void PillButton::leaveEvent(QEvent* /*event*/) {
    hovered_ = false;
    pressed_ = false;
    update();
}

void PillButton::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        pressed_ = true;
        update();
    }
}

void PillButton::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && pressed_) {
        pressed_ = false;
        update();
        if (rect().contains(event->pos())) {
            emit clicked();
        }
    }
}

void PillButton::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    QColor fill;
    QColor border;
    QColor text_color;
    QColor count_color;

    if (active_) {
        fill = pressed_ ? QColor(6, 12, 38) : theme::kInk;
        border = theme::kInk;
        text_color = Qt::white;
        count_color = theme::kAir;
    } else if (pressed_) {
        fill = QColor(234, 242, 255);
        border = theme::kDeep;
        text_color = theme::kDeep;
        count_color = theme::kDeep;
    } else if (hovered_) {
        fill = Qt::white;
        border = theme::kAir;
        text_color = theme::kInk;
        count_color = theme::kSlate;
    } else {
        fill = theme::kPaper;
        border = theme::kLine;
        text_color = theme::kInk;
        count_color = theme::kSlate;
    }

    int y_off = pressed_ ? 1 : 0;
    QRectF r(0.5, 0.5 + y_off, width() - 1.0, height() - 1.0);
    p.setPen(QPen(border, 1.0));
    p.setBrush(fill);
    p.drawRoundedRect(r, 16.0, 16.0);

    // Name: 15px sans 450
    QFont fn = fonts::sans(15, 450, -0.02);
    p.setFont(fn);
    p.setPen(text_color);
    QFontMetrics fmn(fn);
    int nw = fmn.horizontalAdvance(name_);

    // Draw name at x = 14
    int text_y = (height() - fmn.height()) / 2 + fmn.ascent() + y_off;
    p.drawText(14, text_y, name_);

    // Count: 11px mono
    QFont fc = fonts::mono(11, 450);
    p.setFont(fc);
    p.setPen(count_color);
    QFontMetrics fmc(fc);
    QString count_str = QString::number(count_);
    int count_x = 14 + nw + 3;
    int count_y = (height() - fmc.height()) / 2 + fmc.ascent() + y_off;
    p.drawText(count_x, count_y, count_str);
}

} // namespace whalemix::gui
