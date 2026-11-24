#pragma once

#include <QWidget>
#include <QString>
#include <QPainter>

namespace whalemix::gui {

class PillButton : public QWidget {
    Q_OBJECT
public:
    explicit PillButton(const QString& name, int count = 0, QWidget* parent = nullptr);

    void set_name(const QString& name);
    QString name() const { return name_; }

    void set_count(int count);
    int count() const { return count_; }

    void set_active(bool active);
    bool is_active() const { return active_; }

    QSize sizeHint() const override;

signals:
    void clicked();

protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QString name_;
    int count_ = 0;
    bool active_ = false;
    bool hovered_ = false;
    bool pressed_ = false;
};

} // namespace whalemix::gui
