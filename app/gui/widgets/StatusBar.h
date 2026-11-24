#pragma once

#include <QWidget>
#include <QString>

namespace whalemix::gui {

class StatusBar : public QWidget {
    Q_OBJECT
public:
    explicit StatusBar(QWidget* parent = nullptr);

    void set_status(const QString& msg);
    QString status() const { return status_; }

    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString status_ = "Open a Blue A score CSV.";
};

} // namespace whalemix::gui
