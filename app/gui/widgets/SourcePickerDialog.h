#pragma once
#include <QDialog>
#include <QString>
#include <QStringList>
#include <QPoint>

class QLabel;
class QPushButton;
class QFrame;

namespace whalemix::gui {

enum class SourceKind { Csv, Audio };

class SourcePickerDialog : public QDialog {
    Q_OBJECT
public:
    explicit SourcePickerDialog(SourceKind kind = SourceKind::Csv, QWidget* parent = nullptr);

signals:
    void csv_file_chosen(const QString& path);
    void audio_files_chosen(const QStringList& paths);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    void setup_ui();
    void set_kind(SourceKind kind);
    void choose_files();
    void choose_folder();
    void ingest_paths(const QStringList& paths);

    SourceKind kind_;

    QLabel*      title_lbl_         = nullptr;
    QLabel*      subtitle_lbl_      = nullptr;
    QPushButton* tab_csv_btn_       = nullptr;
    QPushButton* tab_audio_btn_     = nullptr;
    QLabel*      drop_icon_label_   = nullptr;
    QLabel*      drop_title_label_  = nullptr;
    QLabel*      drop_hint_label_   = nullptr;
    QFrame*      drop_zone_frame_   = nullptr;
    QPoint       drag_pos_;
};

} // namespace whalemix::gui
