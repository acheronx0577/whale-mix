#pragma once

#include <QWidget>
#include <QString>
#include <QPainter>

namespace whalemix::gui {

class Chip : public QWidget {
    Q_OBJECT
public:
    explicit Chip(const QString& label = "", QWidget* parent = nullptr);

    void set_label(const QString& label);
    QString label() const { return label_; }

    QSize sizeHint() const override;

    static void paint(QPainter* painter, const QRect& rect, const QString& label);
    static QSize size_for_label(const QString& label);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString label_;
};

} // namespace whalemix::gui
