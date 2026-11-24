#include "Kbd.h"
#include "Theme.h"
#include "Fonts.h"
#include <QFontMetrics>

namespace whalemix::gui {

QString Kbd::platform_shortcut(const QString& shortcut) {
#if defined(Q_OS_MACOS)
    QString s = shortcut;
    s.replace("Ctrl", "⌘");
    return s;
#else
    return shortcut;
#endif
}

Kbd::Kbd(const QString& key, KbdStyle style, QWidget* parent)
    : QWidget(parent), key_(key), style_(style) {
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void Kbd::set_key(const QString& key) {
    if (key_ != key) {
        key_ = key;
        updateGeometry();
        update();
    }
}

void Kbd::set_style(KbdStyle style) {
    if (style_ != style) {
        style_ = style;
        updateGeometry();
        update();
    }
}

QSize Kbd::sizeHint() const {
    return size_for_key(key_, style_);
}

QSize Kbd::size_for_key(const QString& key, KbdStyle style) {
    // Fast path for standard keys
    if (key == "Ctrl O" && style == KbdStyle::TopBarSecondary) {
        static QSize s_ctrl_o;
        if (s_ctrl_o.isEmpty()) {
            QString text = platform_shortcut(key);
            QFont f = fonts::mono(10, 450);
            QFontMetrics fm(f);
            s_ctrl_o = QSize(qMax(20, fm.horizontalAdvance(text) + 10), 20);
        }
        return s_ctrl_o;
    }
    if (key == "Ctrl E" && style == KbdStyle::TopBarPrimary) {
        static QSize s_ctrl_e;
        if (s_ctrl_e.isEmpty()) {
            QString text = platform_shortcut(key);
            QFont f = fonts::mono(10, 450);
            QFontMetrics fm(f);
            s_ctrl_e = QSize(qMax(20, fm.horizontalAdvance(text) + 10), 20);
        }
        return s_ctrl_e;
    }
    if (style == KbdStyle::Default) {
        if (key == "C" || key == "N" || key == "R" || key == "J" || key == "K") {
            static QSize s_single_char;
            if (s_single_char.isEmpty()) {
                QFont f = fonts::mono(11, 450);
                QFontMetrics fm(f);
                s_single_char = QSize(qMax(21, fm.horizontalAdvance("M") + 12), 21);
            }
            return s_single_char;
        }
    }

    QString text = platform_shortcut(key);
    int h = (style == KbdStyle::Default) ? 21 : 20;
    int font_px = (style == KbdStyle::Default) ? 11 : 10;
    QFont f = fonts::mono(font_px, 450);
    QFontMetrics fm(f);
    int text_w = fm.horizontalAdvance(text);
    int pad = (style == KbdStyle::Default) ? 6 : 5;
    int w = text_w + pad * 2;
    if (w < h) w = h; // min width = height
    return QSize(w, h);
}

void Kbd::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    paint(&painter, rect(), key_, style_);
}

void Kbd::paint(QPainter* p, const QRect& r, const QString& key, KbdStyle style) {
    if (key.isEmpty()) return;

    p->save();
    p->setRenderHint(QPainter::Antialiasing, true);

    QString text = platform_shortcut(key);
    QColor fill;
    QColor border = Qt::transparent;
    QColor text_color;
    int font_px = 11;

    switch (style) {
    case KbdStyle::Default:
        fill = Qt::white;
        border = theme::kLine;
        text_color = theme::kInk;
        font_px = 11;
        break;
    case KbdStyle::TopBarSecondary:
        fill = theme::kMist;
        border = Qt::transparent;
        text_color = theme::kSlate;
        font_px = 10;
        break;
    case KbdStyle::TopBarPrimary:
        fill = QColor(255, 255, 255, 41); // rgba(255, 255, 255, 0.16)
        border = Qt::transparent;
        text_color = Qt::white;
        font_px = 10;
        break;
    }

    QRectF box_rect(r.x() + 0.5, r.y() + 0.5, r.width() - 1.0, r.height() - 1.0);
    if (border != Qt::transparent) {
        p->setPen(QPen(border, 1.0));
    } else {
        p->setPen(Qt::NoPen);
    }
    p->setBrush(fill);
    p->drawRoundedRect(box_rect, 4.0, 4.0);

    QFont f = fonts::mono(font_px, 450);
    p->setFont(f);
    p->setPen(text_color);
    p->drawText(r, Qt::AlignCenter, text);

    p->restore();
}

} // namespace whalemix::gui
