#include "PillButton.h"
#include "Theme.h"
#include "Fonts.h"
#include <QFontMetrics>
#include <QMouseEvent>
#include <QLocale>

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

namespace {
// Count badge: comma-grouped, same sans family as the name, in a soft rounded counter.
QString count_text(int n) { return QLocale(QLocale::English).toString(n); }
QFont name_font() { return fonts::sans(15, 450, -0.02); }
QFont badge_font() { return fonts::sans(12, 600, 0.0); }
constexpr int kPadL = 14, kGap = 7, kPadR = 6, kBadgeH = 20, kBadgePadX = 7;
}  // namespace

QSize PillButton::sizeHint() const {
    const int nw = QFontMetrics(name_font()).horizontalAdvance(name_);
    const int bw = QFontMetrics(badge_font()).horizontalAdvance(count_text(count_)) + 2 * kBadgePadX;
    return QSize(kPadL + nw + kGap + qMax(bw, kBadgeH) + kPadR, 32);
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

    // Name: 15px sans 450, vertically centred
    QFont fn = name_font();
    p.setFont(fn);
    p.setPen(text_color);
    QFontMetrics fmn(fn);
    const int nw = fmn.horizontalAdvance(name_);
    const int text_y = (height() - fmn.height()) / 2 + fmn.ascent() + y_off;
    p.drawText(kPadL, text_y, name_);

    // Count badge
    QFont fc = badge_font();
    QFontMetrics fmc(fc);
    const QString cs = count_text(count_);
    const int bw = qMax(fmc.horizontalAdvance(cs) + 2 * kBadgePadX, kBadgeH);
    const QRectF badge(kPadL + nw + kGap, (height() - kBadgeH) / 2.0 + y_off, bw, kBadgeH);
    QColor badge_fill = active_ ? QColor(255, 255, 255, 38) : (hovered_ || pressed_ ? theme::kMist : QColor(Qt::white));
    p.setPen(active_ ? Qt::NoPen : QPen(theme::kLine, 1.0));
    p.setBrush(badge_fill);
    p.drawRoundedRect(badge.adjusted(0.5, 0.5, -0.5, -0.5), kBadgeH / 2.0, kBadgeH / 2.0);
    p.setFont(fc);
    p.setPen(active_ ? QColor(Qt::white) : count_color);
    p.drawText(badge, Qt::AlignCenter, cs);
}

} // namespace whalemix::gui
