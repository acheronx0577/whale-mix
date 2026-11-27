#include "SearchBox.h"
#include "Theme.h"
#include "Fonts.h"
#include <QPainter>
#include <QSvgRenderer>
#include <QKeyEvent>
#include <QHBoxLayout>

namespace whalemix::gui {

SearchBox::SearchBox(QWidget* parent) : QWidget(parent) {
    setFixedSize(200, 36);

    line_edit_ = new QLineEdit(this);
    line_edit_->setPlaceholderText("Search WAV files");
    line_edit_->setFont(fonts::body(14));
    line_edit_->setStyleSheet(
        "QLineEdit { border: none; background: transparent; color: #0b1640; padding: 0; }"
        "QLineEdit::placeholder { color: #56658f; }"
    );
    line_edit_->installEventFilter(this);

    // Layout inside: 12px left padding + 17px icon + 8px gap + line_edit + 12px right padding
    line_edit_->setGeometry(37, 2, 200 - 37 - 12, 32);

    connect(line_edit_, &QLineEdit::textChanged, this, &SearchBox::text_changed);
}

QString SearchBox::text() const {
    return line_edit_->text();
}

void SearchBox::clear() {
    line_edit_->clear();
}

void SearchBox::focus_input() {
    line_edit_->setFocus();
    line_edit_->selectAll();
}

bool SearchBox::eventFilter(QObject* watched, QEvent* event) {
    if (watched == line_edit_) {
        if (event->type() == QEvent::FocusIn) {
            is_focused_ = true;
            update();
        } else if (event->type() == QEvent::FocusOut) {
            is_focused_ = false;
            update();
        } else if (event->type() == QEvent::KeyPress) {
            auto* key_event = static_cast<QKeyEvent*>(event);
            if (key_event->key() == Qt::Key_Escape) {
                line_edit_->clearFocus();
                return true;
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

void SearchBox::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    // Background white with border (line or deep if focused), radius 18
    QRectF r(0.5, 0.5, width() - 1.0, height() - 1.0);
    p.setPen(QPen(is_focused_ ? theme::kDeep : theme::kLine, 1.0));
    p.setBrush(Qt::white);
    p.drawRoundedRect(r, 18.0, 18.0);

    // Search icon at x = 12, y = (36 - 17) / 2
    QSvgRenderer svg(QString(":/icons/search.svg"));
    QRect icon_rect(12, (height() - 17) / 2, 17, 17);
    svg.render(&p, icon_rect);
}

void SearchBox::set_placeholder(const QString& text) {
    line_edit_->setPlaceholderText(text);
}

} // namespace whalemix::gui

