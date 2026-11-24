#pragma once

#include <QWidget>
#include <QString>
#include <QPixmap>

namespace whalemix::gui {

class Sidebar : public QWidget {
    Q_OBJECT
public:
    explicit Sidebar(QWidget* parent = nullptr);

    void set_counts(int call, int not_call, int review, int total);
    void set_active_pile(const QString& pile);
    QString active_pile() const { return active_pile_; }

    void set_source_info(const QString& file_name, int num_sounds, int num_pictures);
    void set_has_manual(bool has_manual);

    QSize sizeHint() const override;

signals:
    void pile_filter_changed(const QString& pile);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    QRect pile_row_rect(int index) const; // 0 = Call, 1 = Not-call, 2 = Review, 3 = Total

    int call_count_ = 0;
    int not_call_count_ = 0;
    int review_count_ = 0;
    int total_count_ = 0;

    QString active_pile_ = "all";
    int hovered_row_ = -1;
    int pressed_row_ = -1;

    QString file_name_;
    int num_sounds_ = 0;
    int num_pictures_ = 0;
    bool has_manual_ = false;

    QPixmap dither_pixmap_;
};

} // namespace whalemix::gui
