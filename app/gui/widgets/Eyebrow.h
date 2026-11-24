#pragma once

#include <QWidget>
#include <QString>
#include <QPainter>

namespace whalemix::gui {

class Eyebrow : public QWidget {
    Q_OBJECT
public:
    explicit Eyebrow(const QString& text = "", bool blue = false, QWidget* parent = nullptr);

    void set_text(const QString& text);
    QString text() const { return text_; }

    void set_blue(bool blue);
    bool is_blue() const { return blue_; }

    QSize sizeHint() const override;

    static void paint(QPainter* painter, const QRect& rect, const QString& text, bool blue = false);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString text_;
    bool blue_ = false;
};

} // namespace whalemix::gui
