#include "SourcePickerDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFrame>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QFileInfoList>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QFont>
#include <QStyle>
#include <QSizePolicy>

namespace whalemix::gui {

SourcePickerDialog::SourcePickerDialog(SourceKind kind, QWidget* parent)
    : QDialog(parent), kind_(kind)
{
    setModal(true);
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    setAcceptDrops(true);
    setFixedWidth(560);
    setup_ui();
    set_kind(kind);

    if (parent) {
        QPoint center = parent->mapToGlobal(parent->rect().center());
        move(center.x() - width() / 2, center.y() - height() / 2);
    }
}

void SourcePickerDialog::setup_ui() {
    setStyleSheet(R"(
        QDialog {
            background: #ffffff;
            border: 1px solid #d0dbe9;
            border-radius: 12px;
        }
        QLabel#title {
            font-size: 20px;
            font-weight: 700;
            color: #0b1528;
        }
        QLabel#subtitle {
            font-size: 13px;
            color: #6b7a99;
        }
        QPushButton#tabBtnActive {
            background: #1e3fd8;
            color: #ffffff;
            border: 1px solid #1e3fd8;
            border-radius: 6px;
            padding: 5px 14px;
            font-size: 12px;
            font-weight: 600;
        }
        QPushButton#tabBtnInactive {
            background: #f1f4fc;
            color: #435479;
            border: 1px solid #d9e2f5;
            border-radius: 6px;
            padding: 5px 14px;
            font-size: 12px;
            font-weight: 500;
        }
        QPushButton#tabBtnInactive:hover {
            background: #e3ebfa;
            color: #0b1528;
        }
        QFrame#dropZone {
            background: #f0f4ff;
            border: 1.5px dashed #93aeed;
            border-radius: 10px;
        }
        QLabel#dropTitle {
            font-size: 16px;
            font-weight: 600;
            color: #0b1528;
        }
        QLabel#dropHint {
            font-size: 12px;
            color: #6b7a99;
            font-family: monospace;
        }
        QPushButton#chooseBtn {
            background: #ffffff;
            color: #0b1528;
            border: 1px solid #c8d4f0;
            border-radius: 8px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 500;
        }
        QPushButton#chooseBtn:hover {
            background: #f0f4ff;
            border-color: #93aeed;
        }
        QPushButton#chooseBtn:pressed {
            background: #e1e7f5;
        }
        QPushButton#closeBtn {
            background: transparent;
            border: none;
            color: #6b7a99;
            font-size: 20px;
            font-weight: 300;
            padding: 2px 6px;
        }
        QPushButton#closeBtn:hover {
            color: #0b1528;
        }
    )");

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(28, 24, 28, 24);
    root->setSpacing(0);

    // Header row (title + close button)
    {
        auto* header = new QHBoxLayout();
        header->setSpacing(8);

        auto* title_col = new QVBoxLayout();
        title_col->setSpacing(3);

        title_lbl_ = new QLabel("Open score CSV", this);
        title_lbl_->setObjectName("title");
        title_col->addWidget(title_lbl_);

        subtitle_lbl_ = new QLabel("Blue A score tables \xc2\xb7 accepts .csv", this);
        subtitle_lbl_->setObjectName("subtitle");
        title_col->addWidget(subtitle_lbl_);

        header->addLayout(title_col);
        header->addStretch();

        auto* close_btn = new QPushButton("\xc3\x97", this); // ×
        close_btn->setObjectName("closeBtn");
        close_btn->setFixedSize(28, 28);
        connect(close_btn, &QPushButton::clicked, this, &QDialog::reject);
        header->addWidget(close_btn, 0, Qt::AlignTop);

        root->addLayout(header);
    }

    root->addSpacing(14);

    // Segmented tabs to switch between CSV and Raw Audio
    {
        auto* tabs_row = new QHBoxLayout();
        tabs_row->setSpacing(8);

        tab_csv_btn_ = new QPushButton("Score CSV", this);
        tab_csv_btn_->setObjectName("tabBtnActive");
        connect(tab_csv_btn_, &QPushButton::clicked, this, [this]() {
            set_kind(SourceKind::Csv);
        });
        tabs_row->addWidget(tab_csv_btn_);

        tab_audio_btn_ = new QPushButton("Raw Audio (.wav \xc2\xb7 .flac)", this);
        tab_audio_btn_->setObjectName("tabBtnInactive");
        connect(tab_audio_btn_, &QPushButton::clicked, this, [this]() {
            set_kind(SourceKind::Audio);
        });
        tabs_row->addWidget(tab_audio_btn_);

        tabs_row->addStretch();
        root->addLayout(tabs_row);
    }

    root->addSpacing(14);

    // Drop zone
    {
        drop_zone_frame_ = new QFrame(this);
        drop_zone_frame_->setObjectName("dropZone");
        drop_zone_frame_->setMinimumHeight(180);
        drop_zone_frame_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        auto* dz_layout = new QVBoxLayout(drop_zone_frame_);
        dz_layout->setContentsMargins(24, 28, 24, 28);
        dz_layout->setSpacing(8);
        dz_layout->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

        // Download / arrow down icon
        drop_icon_label_ = new QLabel("\xE2\xAC\x87", drop_zone_frame_); // ⬇
        {
            QFont f = drop_icon_label_->font();
            f.setPixelSize(32);
            drop_icon_label_->setFont(f);
        }
        drop_icon_label_->setStyleSheet("color: #1e3fd8;");
        dz_layout->addWidget(drop_icon_label_, 0, Qt::AlignHCenter);

        // Requested text: "Drop files here"
        drop_title_label_ = new QLabel("Drop files here", drop_zone_frame_);
        drop_title_label_->setObjectName("dropTitle");
        drop_title_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_title_label_, 0, Qt::AlignHCenter);

        drop_hint_label_ = new QLabel("Or choose files or a folder. Accepts .csv.", drop_zone_frame_);
        drop_hint_label_->setObjectName("dropHint");
        drop_hint_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_hint_label_, 0, Qt::AlignHCenter);

        root->addWidget(drop_zone_frame_);
    }

    root->addSpacing(16);

    // Bottom action buttons: Choose files and Choose folder
    {
        auto* btn_row = new QHBoxLayout();
        btn_row->setSpacing(10);
        btn_row->addStretch();

        auto* files_btn = new QPushButton("\xf0\x9f\x93\x84 Choose files", this); // 📄
        files_btn->setObjectName("chooseBtn");
        connect(files_btn, &QPushButton::clicked, this, &SourcePickerDialog::choose_files);
        btn_row->addWidget(files_btn);

        auto* folder_btn = new QPushButton("\xf0\x9f\x93\x81 Choose folder", this); // 📁
        folder_btn->setObjectName("chooseBtn");
        connect(folder_btn, &QPushButton::clicked, this, &SourcePickerDialog::choose_folder);
        btn_row->addWidget(folder_btn);

        root->addLayout(btn_row);
    }
}

void SourcePickerDialog::set_kind(SourceKind kind) {
    kind_ = kind;
    const bool is_csv = (kind == SourceKind::Csv);

    title_lbl_->setText(is_csv ? "Open score CSV" : "Import raw audio");
    subtitle_lbl_->setText(is_csv
        ? "Blue A score tables \xc2\xb7 accepts .csv"
        : "Hydrophone recordings \xc2\xb7 accepts .wav \xc2\xb7 .flac");
    drop_hint_label_->setText(is_csv
        ? "Or choose files or a folder. Accepts .csv."
        : "Or choose files or a folder. Accepts .wav, .flac.");

    tab_csv_btn_->setObjectName(is_csv ? "tabBtnActive" : "tabBtnInactive");
    tab_audio_btn_->setObjectName(is_csv ? "tabBtnInactive" : "tabBtnActive");
    tab_csv_btn_->style()->unpolish(tab_csv_btn_);
    tab_csv_btn_->style()->polish(tab_csv_btn_);
    tab_audio_btn_->style()->unpolish(tab_audio_btn_);
    tab_audio_btn_->style()->polish(tab_audio_btn_);
}

void SourcePickerDialog::choose_files() {
    const bool is_csv = (kind_ == SourceKind::Csv);
    const QString filter = is_csv
        ? "Blue A score CSV (*.csv);;All Files (*.*)"
        : "Audio Files (*.wav *.flac);;All Files (*.*)";

    QStringList files = QFileDialog::getOpenFileNames(
        this,
        is_csv ? "Choose score CSV files" : "Choose audio files",
        QDir::homePath(),
        filter
    );

    if (!files.isEmpty()) {
        ingest_paths(files);
    }
}

void SourcePickerDialog::choose_folder() {
    const bool is_csv = (kind_ == SourceKind::Csv);
    QString folder = QFileDialog::getExistingDirectory(
        this,
        is_csv ? "Choose folder with CSV files" : "Choose folder with audio files",
        QDir::homePath(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    if (folder.isEmpty()) return;

    QStringList name_filters;
    if (is_csv) {
        name_filters << "*.csv";
    } else {
        name_filters << "*.wav" << "*.flac";
    }

    QDir dir(folder);
    QFileInfoList entries = dir.entryInfoList(name_filters, QDir::Files | QDir::Readable, QDir::Name);

    // If active kind filter yielded no files, also check the other extension
    if (entries.isEmpty()) {
        QStringList alt_filters = is_csv ? (QStringList() << "*.wav" << "*.flac") : (QStringList() << "*.csv");
        entries = dir.entryInfoList(alt_filters, QDir::Files | QDir::Readable, QDir::Name);
    }

    QStringList paths;
    for (const QFileInfo& fi : entries) {
        paths << fi.absoluteFilePath();
    }

    if (!paths.isEmpty()) {
        ingest_paths(paths);
    }
}

void SourcePickerDialog::ingest_paths(const QStringList& paths) {
    if (paths.isEmpty()) return;

    QString first_csv;
    QStringList audio_files;

    for (const QString& p : paths) {
        QString ext = "." + QFileInfo(p).suffix().toLower();
        if (ext == ".csv" && first_csv.isEmpty()) {
            first_csv = p;
        } else if (ext == ".wav" || ext == ".flac") {
            audio_files << p;
        }
    }

    // Auto-select files and deliver to MainWindow
    if (!first_csv.isEmpty()) {
        emit csv_file_chosen(first_csv);
    }
    if (!audio_files.isEmpty()) {
        emit audio_files_chosen(audio_files);
    }

    // Auto-close dialog immediately
    accept();
}

void SourcePickerDialog::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    } else {
        event->ignore();
    }
}

void SourcePickerDialog::dropEvent(QDropEvent* event) {
    if (!event->mimeData()->hasUrls()) {
        event->ignore();
        return;
    }

    QStringList paths;
    for (const QUrl& url : event->mimeData()->urls()) {
        if (!url.isLocalFile()) continue;
        QString local = url.toLocalFile();
        QFileInfo fi(local);
        if (fi.isFile()) {
            QString ext = "." + fi.suffix().toLower();
            if (ext == ".csv" || ext == ".wav" || ext == ".flac") {
                paths << local;
            }
        } else if (fi.isDir()) {
            QDir dir(local);
            QFileInfoList entries = dir.entryInfoList(
                QStringList() << "*.csv" << "*.wav" << "*.flac",
                QDir::Files | QDir::Readable, QDir::Name
            );
            for (const QFileInfo& entry : entries) {
                paths << entry.absoluteFilePath();
            }
        }
    }

    if (!paths.isEmpty()) {
        ingest_paths(paths);
        event->acceptProposedAction();
    } else {
        event->ignore();
    }
}

void SourcePickerDialog::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape) {
        reject();
        return;
    }
    QDialog::keyPressEvent(event);
}

void SourcePickerDialog::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        drag_pos_ = event->globalPosition().toPoint() - frameGeometry().topLeft();
        event->accept();
    }
}

void SourcePickerDialog::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        move(event->globalPosition().toPoint() - drag_pos_);
        event->accept();
    }
}

} // namespace whalemix::gui
