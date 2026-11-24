#include "Eyebrow.h"
#include "Theme.h"
#include "Fonts.h"
#include <QFontMetrics>

namespace whalemix::gui {

Eyebrow::Eyebrow(const QString& text, bool blue, QWidget* parent)
    : QWidget(parent), text_(text), blue_(blue) {
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
}

void Eyebrow::set_text(const QString& text) {
    if (text_ != text) {
        text_ = text;
        updateGeometry();
        update();
    }
}

void Eyebrow::set_blue(bool blue) {
    if (blue_ != blue) {
        blue_ = blue;
        update();
    }
}

QSize Eyebrow::sizeHint() const {
    QFont f = fonts::mono(11, 450, 0.1);
    QFontMetrics fm(f);
    int w = fm.horizontalAdvance(text_.toUpper());
    return QSize(w, 14);
}

void Eyebrow::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    paint(&painter, rect(), text_, blue_);
}

void Eyebrow::paint(QPainter* p, const QRect& r, const QString& text, bool blue) {
    if (text.isEmpty()) return;

    p->save();
    p->setRenderHint(QPainter::Antialiasing, true);

    QFont f = fonts::mono(11, 450, 0.1);
    p->setFont(f);
    p->setPen(blue ? theme::kDeep : theme::kSlate);
    p->drawText(r, Qt::AlignVCenter | Qt::AlignLeft, text.toUpper());

    p->restore();
}

} // namespace whalemix::gui
