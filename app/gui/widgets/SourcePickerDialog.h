#pragma once
#include <QDialog>
#include <QString>
#include <QStringList>

class QLabel;
class QPushButton;
class QFrame;

namespace whalemix::gui {

enum class SourceKind { Csv, Audio };

class SourcePickerDialog : public QDialog {
    Q_OBJECT
public:
    explicit SourcePickerDialog(SourceKind kind, QWidget* parent = nullptr);

signals:
    // Emitted only for SourceKind::Csv when a CSV file is chosen
    void csv_file_chosen(const QString& path);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    void setup_ui(SourceKind kind);
    void choose_files(SourceKind kind);
    void choose_folder(SourceKind kind);
    void ingest_paths(const QStringList& paths, SourceKind kind);
    void update_drop_zone();

    SourceKind kind_;
    int queued_count_ = 0;
    qint64 queued_bytes_ = 0;

    QLabel*      drop_icon_label_   = nullptr;
    QLabel*      drop_title_label_  = nullptr;
    QLabel*      drop_hint_label_   = nullptr;
    QLabel*      queued_label_      = nullptr;
    QLabel*      demo_label_        = nullptr;
    QPushButton* clear_btn_         = nullptr;
    QFrame*      drop_zone_frame_   = nullptr;
};

} // namespace whalemix::gui
