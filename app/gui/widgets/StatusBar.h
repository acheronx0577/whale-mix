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
    void set_version_text(const QString& text) { version_ = text; update(); }
    QString version_text() const { return version_; }

    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QString status_ = "Open a Blue A score CSV.";
    QString version_ = QString::fromUtf8("v0.2 \xc2\xb7 BLUE A + LAYA");
};

} // namespace whalemix::gui
