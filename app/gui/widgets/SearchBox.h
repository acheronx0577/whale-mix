#pragma once

#include <QWidget>
#include <QLineEdit>
#include <QString>

namespace whalemix::gui {

class SearchBox : public QWidget {
    Q_OBJECT
public:
    explicit SearchBox(QWidget* parent = nullptr);

    QString text() const;
    void clear();
    void focus_input();
    void set_placeholder(const QString& text);

signals:
    void text_changed(const QString& text);

protected:
    void paintEvent(QPaintEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QLineEdit* line_edit_ = nullptr;
    bool is_focused_ = false;
};

} // namespace whalemix::gui
