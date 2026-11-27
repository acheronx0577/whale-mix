#include "StatusBar.h"
#include "Theme.h"
#include "Fonts.h"
#include <QPainter>
#include <QFontMetrics>

namespace whalemix::gui {

StatusBar::StatusBar(QWidget* parent) : QWidget(parent) {
    setFixedHeight(36);
}

void StatusBar::set_status(const QString& msg) {
    if (status_ != msg) {
        status_ = msg;
        update();
    }
}

QSize StatusBar::sizeHint() const {
    return QSize(1440, 36);
}

void StatusBar::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    // Background white
    p.fillRect(0, 1, width(), 35, Qt::white);

    // Top border: 1px line
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(0, 0, width(), 0);

    // Right version text: "v0.2 · BLUE A + LAYA" or "v0.2 · WHALEMIX LIGHT" (mono 11px, .1em, slate)
    QFont ver_font = fonts::mono(11, 450, 0.1);
    p.setFont(ver_font);
    p.setPen(theme::kSlate);
    QFontMetrics fm_ver(ver_font);
    const QString& ver_text = version_;
    int ver_w = fm_ver.horizontalAdvance(ver_text);
    int ver_x = width() - 24 - ver_w;
    int ver_y = 1 + (35 - fm_ver.height()) / 2 + fm_ver.ascent();
    p.drawText(ver_x, ver_y, ver_text);

    // Left dot: 6px sky circle at x = 24
    int dot_y = 1 + (35 - 6) / 2;
    p.setPen(Qt::NoPen);
    p.setBrush(theme::kSky);
    p.drawEllipse(QRectF(24.0, dot_y, 6.0, 6.0));

    // Left status text: mono 13px slate, 12px gap after dot
    int text_x = 24 + 6 + 12;
    int max_text_w = ver_x - 16 - text_x;
    if (max_text_w > 10) {
        QFont status_font = fonts::mono(13, 450);
        p.setFont(status_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_status(status_font);
        QString elided = fm_status.elidedText(status_, Qt::ElideRight, max_text_w);
        int text_y = 1 + (35 - fm_status.height()) / 2 + fm_status.ascent();
        p.drawText(text_x, text_y, elided);
    }
}

} // namespace whalemix::gui
