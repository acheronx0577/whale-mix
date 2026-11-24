#pragma once

#include <QWidget>
#include <QString>
#include <QPainter>

namespace whalemix::gui {

enum class KbdStyle {
    Default,         // White fill, line border, ink text, min 21x21, radius 4
    TopBarSecondary, // Mist fill, no border, slate text, 20px high, radius 4
    TopBarPrimary    // rgba(255,255,255,0.16) fill, no border, white text, 20px high, radius 4
};

class Kbd : public QWidget {
    Q_OBJECT
public:
    explicit Kbd(const QString& key = "", KbdStyle style = KbdStyle::Default, QWidget* parent = nullptr);

    void set_key(const QString& key);
    QString key() const { return key_; }

    void set_style(KbdStyle style);
    KbdStyle style() const { return style_; }

    QSize sizeHint() const override;

    static QString platform_shortcut(const QString& shortcut);
    static void paint(QPainter* painter, const QRect& rect, const QString& key, KbdStyle style);
    static QSize size_for_key(const QString& key, KbdStyle style);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString key_;
    KbdStyle style_ = KbdStyle::Default;
};

} // namespace whalemix::gui
