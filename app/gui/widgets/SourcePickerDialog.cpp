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
#include <QDialogButtonBox>
#include <QIcon>
#include <QFont>
#include <QSizePolicy>
#include <QSpacerItem>
#include <QApplication>
#include <QScreen>

namespace whalemix::gui {

namespace {

// Format bytes to a human-readable size string
QString format_bytes(qint64 bytes) {
    if (bytes < 1024)
        return QString("%1 B").arg(bytes);
    if (bytes < 1024 * 1024)
        return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
}

// SVG-based download/arrow-down icon as inline data URI (simple, no file dep)
constexpr const char* kDownloadSvg =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40\" height=\"40\" viewBox=\"0 0 24 24\" fill=\"none\">"
    "<path d=\"M12 3v13M7 11l5 5 5-5\" stroke=\"#1e3fd8\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
    "<path d=\"M4 20h16\" stroke=\"#1e3fd8\" stroke-width=\"2\" stroke-linecap=\"round\"/>"
    "</svg>";

} // namespace

SourcePickerDialog::SourcePickerDialog(SourceKind kind, QWidget* parent)
    : QDialog(parent), kind_(kind)
{
    setModal(true);
    setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    setAcceptDrops(true);
    setFixedWidth(560);
    setup_ui(kind);

    // Center over parent
    if (parent) {
        QPoint center = parent->mapToGlobal(parent->rect().center());
        move(center.x() - width() / 2, center.y() - height() / 2);
    }
}

void SourcePickerDialog::setup_ui(SourceKind kind) {
    // ── Titles ────────────────────────────────────────────────────────────
    const bool is_csv = (kind == SourceKind::Csv);
    const QString title    = is_csv ? "Open score CSV" : "Import raw audio";
    const QString subtitle = is_csv
        ? "Blue A score tables \xc2\xb7 accepts .csv"
        : "Hydrophone recordings \xc2\xb7 accepts .wav \xc2\xb7 .flac";
    const QString drop_hint_text = is_csv
        ? "Or choose files or a folder. Accepts .csv."
        : "Or choose files or a folder. Accepts .wav, .flac.";
    const QString files_filter = is_csv
        ? "Blue A score CSV (*.csv)"
        : "Audio Files (*.wav *.flac)";

    // ── Outer layout ──────────────────────────────────────────────────────
    setStyleSheet(R"(
        QDialog {
            background: #ffffff;
            border: 1px solid #e1e7f5;
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
        QFrame#dropZone {
            background: #f0f4ff;
            border: 1.5px dashed #93aeed;
            border-radius: 10px;
        }
        QLabel#dropTitle {
            font-size: 15px;
            font-weight: 600;
            color: #0b1528;
        }
        QLabel#dropHint {
            font-size: 12px;
            color: #6b7a99;
            font-family: monospace;
        }
        QLabel#queuedLabel {
            font-size: 13px;
            font-weight: 600;
            color: #0b1528;
        }
        QLabel#demoLabel {
            font-size: 12px;
            color: #6b7a99;
        }
        QPushButton#chooseBtn {
            background: #ffffff;
            color: #0b1528;
            border: 1px solid #c8d4f0;
            border-radius: 8px;
            padding: 8px 16px;
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
        QPushButton#clearBtn {
            background: transparent;
            color: #6b7a99;
            border: none;
            font-size: 12px;
            padding: 4px 8px;
        }
        QPushButton#clearBtn:hover {
            color: #0b1528;
        }
        QPushButton#closeBtn {
            background: transparent;
            border: none;
            color: #6b7a99;
            font-size: 18px;
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

    // ── Header row (title + × close) ──────────────────────────────────────
    {
        auto* header = new QHBoxLayout();
        header->setSpacing(8);

        auto* title_col = new QVBoxLayout();
        title_col->setSpacing(2);

        auto* title_lbl = new QLabel(title, this);
        title_lbl->setObjectName("title");
        title_col->addWidget(title_lbl);

        auto* sub_lbl = new QLabel(subtitle, this);
        sub_lbl->setObjectName("subtitle");
        title_col->addWidget(sub_lbl);

        header->addLayout(title_col);
        header->addStretch();

        auto* close_btn = new QPushButton("\xc3\x97", this);  // × (UTF-8 U+00D7)
        close_btn->setObjectName("closeBtn");
        close_btn->setFixedSize(28, 28);
        connect(close_btn, &QPushButton::clicked, this, &QDialog::reject);
        header->addWidget(close_btn, 0, Qt::AlignTop);

        root->addLayout(header);
    }

    root->addSpacing(20);

    // ── Drop zone ─────────────────────────────────────────────────────────
    {
        drop_zone_frame_ = new QFrame(this);
        drop_zone_frame_->setObjectName("dropZone");
        drop_zone_frame_->setMinimumHeight(180);
        drop_zone_frame_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        auto* dz_layout = new QVBoxLayout(drop_zone_frame_);
        dz_layout->setContentsMargins(24, 24, 24, 24);
        dz_layout->setSpacing(6);
        dz_layout->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

        // Download icon (SVG rendered via QLabel with rich text)
        drop_icon_label_ = new QLabel(this);
        drop_icon_label_->setAlignment(Qt::AlignHCenter);
        drop_icon_label_->setText(QString::fromUtf8(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40\" height=\"40\" viewBox=\"0 0 24 24\" fill=\"none\">"
            "<path d=\"M12 3v13\" stroke=\"#1e3fd8\" stroke-width=\"2\" stroke-linecap=\"round\"/>"
            "<path d=\"M7 11l5 5 5-5\" stroke=\"#1e3fd8\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>"
            "<path d=\"M4 20h16\" stroke=\"#1e3fd8\" stroke-width=\"2\" stroke-linecap=\"round\"/>"
            "</svg>"));
        // Use Unicode arrow as fallback if SVG not rendered
        drop_icon_label_->setText("\xE2\xAC\x87");  // ⬇ U+2B07
        {
            QFont f = drop_icon_label_->font();
            f.setPixelSize(32);
            drop_icon_label_->setFont(f);
        }
        drop_icon_label_->setStyleSheet("color: #1e3fd8;");
        dz_layout->addWidget(drop_icon_label_, 0, Qt::AlignHCenter);

        drop_title_label_ = new QLabel("Drop 100s or 1000s of files here", drop_zone_frame_);
        drop_title_label_->setObjectName("dropTitle");
        drop_title_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_title_label_, 0, Qt::AlignHCenter);

        drop_hint_label_ = new QLabel(drop_hint_text, drop_zone_frame_);
        drop_hint_label_->setObjectName("dropHint");
        drop_hint_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_hint_label_, 0, Qt::AlignHCenter);

        // Queued count — hidden initially
        queued_label_ = new QLabel("", drop_zone_frame_);
        queued_label_->setObjectName("queuedLabel");
        queued_label_->setAlignment(Qt::AlignHCenter);
        queued_label_->hide();
        dz_layout->addWidget(queued_label_, 0, Qt::AlignHCenter);

        demo_label_ = new QLabel("Demo only \xe2\x80\x94 nothing is uploaded or opened.", drop_zone_frame_);
        demo_label_->setObjectName("demoLabel");
        demo_label_->setAlignment(Qt::AlignHCenter);
        demo_label_->hide();
        dz_layout->addWidget(demo_label_, 0, Qt::AlignHCenter);

        root->addWidget(drop_zone_frame_);
    }

    root->addSpacing(16);

    // ── Bottom row: Clear (left) + Choose buttons (right) ─────────────────
    {
        auto* btn_row = new QHBoxLayout();
        btn_row->setSpacing(8);

        clear_btn_ = new QPushButton("Clear", this);
        clear_btn_->setObjectName("clearBtn");
        clear_btn_->hide();
        connect(clear_btn_, &QPushButton::clicked, this, [this]() {
            queued_count_ = 0;
            queued_bytes_ = 0;
            update_drop_zone();
        });
        btn_row->addWidget(clear_btn_);

        btn_row->addStretch();

        auto* files_btn = new QPushButton("\xf0\x9f\x93\x84 Choose files", this);  // 📄
        files_btn->setObjectName("chooseBtn");
        connect(files_btn, &QPushButton::clicked, this, [this, kind]() {
            choose_files(kind);
        });
        btn_row->addWidget(files_btn);

        auto* folder_btn = new QPushButton("\xf0\x9f\x93\x81 Choose folder", this);  // 📁
        folder_btn->setObjectName("chooseBtn");
        connect(folder_btn, &QPushButton::clicked, this, [this, kind]() {
            choose_folder(kind);
        });
        btn_row->addWidget(folder_btn);

        root->addLayout(btn_row);
    }
}

void SourcePickerDialog::choose_files(SourceKind kind) {
    const bool is_csv = (kind == SourceKind::Csv);
    const QString filter = is_csv
        ? "Blue A score CSV (*.csv)"
        : "Audio Files (*.wav *.flac)";

    QStringList files = QFileDialog::getOpenFileNames(
        this,
        is_csv ? "Choose score CSV files" : "Choose audio files",
        QDir::homePath(),
        filter
    );

    if (!files.isEmpty()) {
        ingest_paths(files, kind);
    }
}

void SourcePickerDialog::choose_folder(SourceKind kind) {
    const bool is_csv = (kind == SourceKind::Csv);
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

    QStringList paths;
    for (const QFileInfo& fi : entries) {
        paths << fi.absoluteFilePath();
    }

    if (!paths.isEmpty()) {
        ingest_paths(paths, kind);
    }
}

void SourcePickerDialog::ingest_paths(const QStringList& paths, SourceKind kind) {
    for (const QString& p : paths) {
        QFileInfo fi(p);
        queued_count_++;
        queued_bytes_ += fi.size();

        // For CSV kind, emit so MainWindow can actually open the first file
        if (kind == SourceKind::Csv && queued_count_ == 1) {
            emit csv_file_chosen(p);
        }
    }
    update_drop_zone();
}

void SourcePickerDialog::update_drop_zone() {
    if (queued_count_ > 0) {
        queued_label_->setText(
            QString("%1 file%2 \xc2\xb7 %3 queued")
                .arg(queued_count_)
                .arg(queued_count_ == 1 ? "" : "s")
                .arg(format_bytes(queued_bytes_))
        );
        queued_label_->show();
        demo_label_->show();
        clear_btn_->show();

        // Dim the drop zone slightly to show it has content
        drop_zone_frame_->setStyleSheet(
            "QFrame#dropZone {"
            "  background: #e8eeff;"
            "  border: 1.5px dashed #1e3fd8;"
            "  border-radius: 10px;"
            "}"
        );
    } else {
        queued_label_->hide();
        demo_label_->hide();
        clear_btn_->hide();
        drop_zone_frame_->setStyleSheet("");  // reset to global stylesheet
    }
}

// ── Drag & drop ────────────────────────────────────────────────────────────

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

    const bool is_csv = (kind_ == SourceKind::Csv);
    QStringList accepted_ext = is_csv
        ? QStringList{".csv"}
        : QStringList{".wav", ".flac"};

    QStringList paths;
    for (const QUrl& url : event->mimeData()->urls()) {
        if (!url.isLocalFile()) continue;
        QString local = url.toLocalFile();
        QFileInfo fi(local);
        if (fi.isFile()) {
            QString ext = "." + fi.suffix().toLower();
            if (accepted_ext.contains(ext)) {
                paths << local;
            }
        } else if (fi.isDir()) {
            QStringList filters;
            for (const QString& e : accepted_ext)
                filters << ("*" + e);
            QDir dir(local);
            QFileInfoList entries = dir.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::Name);
            for (const QFileInfo& entry : entries)
                paths << entry.absoluteFilePath();
        }
    }

    if (!paths.isEmpty()) {
        ingest_paths(paths, kind_);
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

} // namespace whalemix::gui
