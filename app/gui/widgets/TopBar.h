#pragma once

#include <QWidget>
#include <QString>
#include <QRect>

namespace whalemix::gui {

class TopBar : public QWidget {
    Q_OBJECT
public:
    explicit TopBar(QWidget* parent = nullptr);

    void set_file_name(const QString& name);
    QString file_name() const { return file_name_; }

    QSize sizeHint() const override;

    QRect open_button_rect() const;
    QRect export_button_rect() const;
    QRect min_button_rect() const;
    QRect max_button_rect() const;
    QRect close_button_rect() const;

signals:
    void open_requested();
    void import_audio_requested();
    void export_requested();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
private:
    QString file_name_;
    bool open_hovered_ = false;
    bool export_hovered_ = false;
    bool min_hovered_ = false;
    bool max_hovered_ = false;
    bool close_hovered_ = false;

    bool open_pressed_ = false;
    bool export_pressed_ = false;
    int control_pressed_ = -1; // 0 = min, 1 = max, 2 = close
};

} // namespace whalemix::gui
