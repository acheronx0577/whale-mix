#include "TopBar.h"
#include "Theme.h"
#include "Fonts.h"
#include "Kbd.h"
#include <QPainter>
#include <QMouseEvent>
#include <QSvgRenderer>
#include <QFontMetrics>
#include <QWindow>
#include <QMenu>
#include <QAction>

namespace whalemix::gui {

TopBar::TopBar(QWidget* parent) : QWidget(parent) {
    setFixedHeight(72);
    setMouseTracking(true);
}

void TopBar::set_file_name(const QString& name) {
    if (file_name_ != name) {
        file_name_ = name;
        update();
    }
}

QSize TopBar::sizeHint() const {
    return QSize(1440, 72);
}

QRect TopBar::min_button_rect() const {
    int btn_w = 46;
    return QRect(width() - btn_w * 3, 0, btn_w, 71);
}

QRect TopBar::max_button_rect() const {
    int btn_w = 46;
    return QRect(width() - btn_w * 2, 0, btn_w, 71);
}

QRect TopBar::close_button_rect() const {
    int btn_w = 46;
    return QRect(width() - btn_w, 0, btn_w, 71);
}

QRect TopBar::export_button_rect() const {
    int y = (71 - 46) / 2;
    int exp_x = width() - (46 * 3) - 16 - 220;
    return QRect(exp_x, y, 220, 46);
}

QRect TopBar::open_button_rect() const {
    int y = (71 - 46) / 2;
    int exp_x = width() - (46 * 3) - 16 - 220;
    int open_x = exp_x - 16 - 180;
    return QRect(open_x, y, 180, 46);
}

void TopBar::mouseMoveEvent(QMouseEvent* event) {
    bool on_open = open_button_rect().contains(event->pos());
    bool on_export = export_button_rect().contains(event->pos());
    bool on_min = min_button_rect().contains(event->pos());
    bool on_max = max_button_rect().contains(event->pos());
    bool on_close = close_button_rect().contains(event->pos());

    if (on_open != open_hovered_ ||
        on_export != export_hovered_ ||
        on_min != min_hovered_ ||
        on_max != max_hovered_ ||
        on_close != close_hovered_) {
        open_hovered_ = on_open;
        export_hovered_ = on_export;
        min_hovered_ = on_min;
        max_hovered_ = on_max;
        close_hovered_ = on_close;

        if (open_hovered_ || export_hovered_) {
            setCursor(Qt::PointingHandCursor);
        } else {
            setCursor(Qt::ArrowCursor);
        }
        update();
    }
}

void TopBar::leaveEvent(QEvent* /*event*/) {
    if (open_hovered_ || export_hovered_ || min_hovered_ || max_hovered_ || close_hovered_ ||
        open_pressed_ || export_pressed_ || control_pressed_ >= 0) {
        open_hovered_ = false;
        export_hovered_ = false;
        min_hovered_ = false;
        max_hovered_ = false;
        close_hovered_ = false;
        open_pressed_ = false;
        export_pressed_ = false;
        control_pressed_ = -1;
        setCursor(Qt::ArrowCursor);
        update();
    }
}

void TopBar::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (min_button_rect().contains(event->pos())) {
            control_pressed_ = 0;
            update();
        } else if (max_button_rect().contains(event->pos())) {
            control_pressed_ = 1;
            update();
        } else if (close_button_rect().contains(event->pos())) {
            control_pressed_ = 2;
            update();
        } else if (open_button_rect().contains(event->pos())) {
            open_pressed_ = true;
            update();
        } else if (export_button_rect().contains(event->pos())) {
            export_pressed_ = true;
            update();
        } else {
            // Drag the window via system native move
            if (window() && window()->windowHandle()) {
                window()->windowHandle()->startSystemMove();
            }
        }
    }
}

void TopBar::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (open_pressed_) {
            open_pressed_ = false;
            update();
            if (open_button_rect().contains(event->pos())) {
                show_source_menu();
            }
            return;
        }
        if (export_pressed_) {
            export_pressed_ = false;
            update();
            if (export_button_rect().contains(event->pos())) {
                emit export_requested();
            }
            return;
        }
        if (control_pressed_ >= 0) {
            int pressed = control_pressed_;
            control_pressed_ = -1;
            update();
            if (window()) {
                if (pressed == 0 && min_button_rect().contains(event->pos())) {
                    window()->showMinimized();
                } else if (pressed == 1 && max_button_rect().contains(event->pos())) {
                    if (window()->isMaximized()) {
                        window()->showNormal();
                    } else {
                        window()->showMaximized();
                    }
                } else if (pressed == 2 && close_button_rect().contains(event->pos())) {
                    window()->close();
                }
            }
            return;
        }
    }
}

void TopBar::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (!open_button_rect().contains(event->pos()) &&
            !export_button_rect().contains(event->pos()) &&
            !min_button_rect().contains(event->pos()) &&
            !max_button_rect().contains(event->pos()) &&
            !close_button_rect().contains(event->pos())) {
            if (window()) {
                if (window()->isMaximized()) {
                    window()->showNormal();
                } else {
                    window()->showMaximized();
                }
                update();
            }
        }
    }
}

void TopBar::show_source_menu() {
    QMenu menu(this);
    menu.setStyleSheet(
        "QMenu {"
        "  background-color: #ffffff;"
        "  border: 1px solid #e1e7f5;"
        "  border-radius: 8px;"
        "  padding: 6px;"
        "  font-family: 'DM Sans', -apple-system, sans-serif;"
        "  font-size: 13px;"
        "}"
        "QMenu::item {"
        "  padding: 8px 18px 8px 12px;"
        "  border-radius: 6px;"
        "  color: #0b1528;"
        "}"
        "QMenu::item:selected {"
        "  background-color: #f1f4fc;"
        "  color: #1e3fd8;"
        "}"
    );

    auto* open_csv = menu.addAction("Open score CSV…");
    open_csv->setShortcut(QKeySequence("Ctrl+O"));
    connect(open_csv, &QAction::triggered, this, &TopBar::open_requested);

    auto* import_audio = menu.addAction("Import raw audio (.wav · .flac)…");
    connect(import_audio, &QAction::triggered, this, &TopBar::import_audio_requested);

    QRect r = open_button_rect();
    QPoint pt = mapToGlobal(QPoint(r.x(), r.bottom() + 4));
    menu.exec(pt);
}

void TopBar::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    // Background
    p.fillRect(0, 0, width(), 71, Qt::white);

    // Bottom border 1px line
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(0, 71, width(), 71);

    // Left brand mark: 36x36 pixel whale logo at x = 24, vertically centered in 71px bar
    static QPixmap logo_pixmap;
    if (logo_pixmap.isNull()) {
        logo_pixmap.load(":/icons/whale-logo-native-53.png");
    }
    int brand_y = (71 - 36) / 2;
    if (!logo_pixmap.isNull()) {
        p.drawPixmap(QRect(24, brand_y, 36, 36), logo_pixmap);
    }

    // WHALE MIX text: mono 11px 450 weight .12em ink with 10px gap
    QFont brand_font = fonts::mono(11, 450, 0.12);
    p.setFont(brand_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_brand(brand_font);
    int brand_text_x = 24 + 36 + 10; // 10px gap
    int text_y = (71 - fm_brand.height()) / 2 + fm_brand.ascent();
    p.drawText(brand_text_x, text_y, "WHALE MIX");

    int cur_x = brand_text_x + fm_brand.horizontalAdvance("WHALE MIX");

    // File name crumb if not empty
    if (!file_name_.isEmpty()) {
        // Separator "/" with 10px left margin, 12px right margin
        QFont sep_font = fonts::mono(13, 450);
        p.setFont(sep_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_sep(sep_font);
        int sep_text_y = (71 - fm_sep.height()) / 2 + fm_sep.ascent();
        p.drawText(cur_x + 10, sep_text_y, "/");
        cur_x += 10 + fm_sep.horizontalAdvance("/") + 12;

        // Elided file name: mono 13px slate
        int open_x = open_button_rect().x();
        int max_w = open_x - 16 - cur_x;
        if (max_w > 20) {
            QFont file_font = fonts::mono(13, 450);
            p.setFont(file_font);
            QFontMetrics fm_file(file_font);
            QString elided = fm_file.elidedText(file_name_, Qt::ElideRight, max_w);
            int file_y = (71 - fm_file.height()) / 2 + fm_file.ascent();
            p.drawText(cur_x, file_y, elided);
        }
    }

    // Open / New source button: 180 x 46, radius 8, white fill, line border (hover paper)
    QRect r_open = open_button_rect();
    if (open_pressed_) {
        r_open = r_open.adjusted(0, 1, 0, 1);
    }
    QRectF r_open_f(r_open.x() + 0.5, r_open.y() + 0.5, r_open.width() - 1.0, r_open.height() - 1.0);
    p.setPen(QPen(open_pressed_ ? theme::kDeep : theme::kLine, 1.0));
    p.setBrush(open_pressed_ ? QColor(238, 243, 252) : (open_hovered_ ? theme::kPaper : Qt::white));
    p.drawRoundedRect(r_open_f, 8.0, 8.0);

    // Plus icon: mono 15px 600 ink
    QFont plus_font = fonts::mono(15, 600);
    p.setFont(plus_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_plus(plus_font);
    int plus_x = r_open.x() + 15;
    int plus_y = r_open.y() + (46 - fm_plus.height()) / 2 + fm_plus.ascent() - 1;
    p.drawText(plus_x, plus_y, "+");

    // Text "New source": 14px sans 450 ink, 8px gap after plus
    QFont btn_font = fonts::sans(14, 450, -0.02);
    p.setFont(btn_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_btn(btn_font);
    int text_open_x = plus_x + fm_plus.horizontalAdvance("+") + 8;
    int text_open_y = r_open.y() + (46 - fm_btn.height()) / 2 + fm_btn.ascent();
    p.drawText(text_open_x, text_open_y, "New source");

    // Kbd on right: 7px right pad
    QSize kbd_open_sz = Kbd::size_for_key("Ctrl O", KbdStyle::TopBarSecondary);
    QRect kbd_open_rect(r_open.right() - 7 - kbd_open_sz.width(),
                        r_open.y() + (46 - kbd_open_sz.height()) / 2,
                        kbd_open_sz.width(), kbd_open_sz.height());
    Kbd::paint(&p, kbd_open_rect, "Ctrl O", KbdStyle::TopBarSecondary);

    // Export button: 220 x 46, radius 8, deep fill (hover primaryHover), white text
    QRect r_exp = export_button_rect();
    if (export_pressed_) {
        r_exp = r_exp.adjusted(0, 1, 0, 1);
    }
    QRectF r_exp_f(r_exp.x() + 0.5, r_exp.y() + 0.5, r_exp.width() - 1.0, r_exp.height() - 1.0);
    p.setPen(QPen(theme::kDeep, 1.0));
    p.setBrush(export_pressed_ ? QColor(15, 36, 130) : (export_hovered_ ? theme::kPrimaryHover : theme::kDeep));
    p.drawRoundedRect(r_exp_f, 8.0, 8.0);

    // Download icon: 17x17 at padding 17 left (cached pixmap for zero-overhead repaints)
    static QPixmap exp_pixmap;
    if (exp_pixmap.isNull()) {
        exp_pixmap = QPixmap(17, 17);
        exp_pixmap.fill(Qt::transparent);
        QPainter p_icon(&exp_pixmap);
        p_icon.setRenderHint(QPainter::Antialiasing, true);
        QSvgRenderer svg(QString(":/icons/download.svg"));
        if (svg.isValid()) {
            svg.render(&p_icon, QRectF(0, 0, 17, 17));
        }
    }
    QRect icon_exp_rect(r_exp.x() + 17, r_exp.y() + (46 - 17) / 2, 17, 17);
    if (!exp_pixmap.isNull()) {
        p.drawPixmap(icon_exp_rect, exp_pixmap);
    }

    // Text "Export report": 14px sans 500 white
    p.setFont(btn_font);
    p.setPen(Qt::white);
    int text_exp_x = icon_exp_rect.right() + 10;
    int text_exp_y = r_exp.y() + (46 - fm_btn.height()) / 2 + fm_btn.ascent();
    p.drawText(text_exp_x, text_exp_y, "Export report");

    // Kbd on right: 7px right pad
    QSize kbd_exp_sz = Kbd::size_for_key("Ctrl E", KbdStyle::TopBarPrimary);
    QRect kbd_exp_rect(r_exp.right() - 7 - kbd_exp_sz.width(),
                       r_exp.y() + (46 - kbd_exp_sz.height()) / 2,
                       kbd_exp_sz.width(), kbd_exp_sz.height());
    Kbd::paint(&p, kbd_exp_rect, "Ctrl E", KbdStyle::TopBarPrimary);

    // Window controls at top right: 3 buttons (Minimize, Maximize/Restore, Close)
    // 1. Minimize button
    QRect r_min = min_button_rect();
    if (control_pressed_ == 0) {
        p.fillRect(r_min, QColor(0, 0, 0, 24));
    } else if (min_hovered_) {
        p.fillRect(r_min, QColor(0, 0, 0, 14));
    }
    int cx_min = r_min.center().x();
    int cy_min = r_min.center().y() + (control_pressed_ == 0 ? 1 : 0);
    p.setPen(QPen(theme::kInk, 1.25));
    p.drawLine(cx_min - 5, cy_min, cx_min + 5, cy_min);

    // 2. Maximize / Restore button
    QRect r_max = max_button_rect();
    if (control_pressed_ == 1) {
        p.fillRect(r_max, QColor(0, 0, 0, 24));
    } else if (max_hovered_) {
        p.fillRect(r_max, QColor(0, 0, 0, 14));
    }
    int cx_max = r_max.center().x();
    int cy_max = r_max.center().y() + (control_pressed_ == 1 ? 1 : 0);
    bool is_max = window() && window()->isMaximized();
    p.setPen(QPen(theme::kInk, 1.25));
    if (!is_max) {
        p.setBrush(Qt::NoBrush);
        p.drawRect(cx_max - 5, cy_max - 5, 10, 10);
    } else {
        p.setBrush(Qt::NoBrush);
        p.drawRect(cx_max - 3, cy_max - 6, 8, 8);
        p.setBrush((control_pressed_ == 1 || max_hovered_) ? QColor(240, 240, 240) : Qt::white);
        p.drawRect(cx_max - 6, cy_max - 3, 8, 8);
    }

    // 3. Close button
    QRect r_close = close_button_rect();
    if (control_pressed_ == 2) {
        p.fillRect(r_close, QColor(196, 15, 31));
        p.setPen(QPen(Qt::white, 1.25));
    } else if (close_hovered_) {
        p.fillRect(r_close, QColor(232, 17, 35));
        p.setPen(QPen(Qt::white, 1.25));
    } else {
        p.setPen(QPen(theme::kInk, 1.25));
    }
    int cx_close = r_close.center().x();
    int cy_close = r_close.center().y() + (control_pressed_ == 2 ? 1 : 0);
    p.drawLine(cx_close - 5, cy_close - 5, cx_close + 5, cy_close + 5);
    p.drawLine(cx_close - 5, cy_close + 5, cx_close + 5, cy_close - 5);
}

} // namespace whalemix::gui
