#include "SourcePickerDialog.h"
#include "Theme.h"
#include "Fonts.h"

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
#include <QStyle>
#include <QPainter>
#include <QSvgRenderer>
#include <QPixmap>
#include <QIcon>

namespace whalemix::gui {

namespace {

constexpr const char* kDropSvg =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"36\" height=\"36\" viewBox=\"0 0 24 24\" fill=\"none\" "
    "stroke=\"#1e3fd8\" stroke-width=\"1.75\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    "<path d=\"M12 3v10\"/>"
    "<path d=\"m8 9 4 4 4-4\"/>"
    "<path d=\"M5 16v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3\"/>"
    "</svg>";

constexpr const char* kFilesSvg =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" "
    "stroke=\"#0b1640\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    "<path d=\"M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z\"/>"
    "<path d=\"M14 3v5h5\"/>"
    "</svg>";

constexpr const char* kFolderSvg =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" "
    "stroke=\"#0b1640\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    "<path d=\"M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z\"/>"
    "</svg>";

constexpr const char* kCloseSvg =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" "
    "stroke=\"#56658f\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    "<line x1=\"18\" y1=\"6\" x2=\"6\" y2=\"18\"/>"
    "<line x1=\"6\" y1=\"6\" x2=\"18\" y2=\"18\"/>"
    "</svg>";

QPixmap render_svg(const char* svg_xml, int w, int h) {
    QPixmap pix(w * 2, h * 2);
    pix.fill(Qt::transparent);
    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing, true);
    QByteArray bytes(svg_xml);
    QSvgRenderer renderer(bytes);
    renderer.render(&p);
    pix.setDevicePixelRatio(2.0);
    return pix;
}

} // namespace

SourcePickerDialog::SourcePickerDialog(SourceKind kind, QWidget* parent)
    : QDialog(parent), kind_(kind)
{
    setModal(true);
    // Frameless window: zero native OS title bar, zero double close buttons
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    setAttribute(Qt::WA_TranslucentBackground, false);
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
            background-color: #ffffff;
            border: 1px solid #d9e4f7;
            border-radius: 12px;
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
            color: #0b1640;
        }
        QFrame#dropZone {
            background: #f6f8ff;
            border: 1.5px dashed #1e3fd8;
            border-radius: 10px;
        }
        QPushButton#actionBtn {
            background: #ffffff;
            color: #0b1640;
            border: 1px solid #d9e4f7;
            border-radius: 8px;
            padding: 0 16px;
            height: 42px;
            min-height: 42px;
            font-size: 13px;
            font-weight: 500;
        }
        QPushButton#actionBtn:hover {
            background: #f6f8ff;
            border-color: #1e3fd8;
        }
        QPushButton#actionBtn:pressed {
            background: #eaf2ff;
        }
        QPushButton#closeBtn {
            background: transparent;
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 32px;
            min-width: 32px;
            min-height: 32px;
        }
        QPushButton#closeBtn:hover {
            background: #f1f4fc;
        }
    )");

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(24, 24, 24, 24);
    root->setSpacing(16);

    // ── Header (Title, Subtitle, and single sleek × Close Button) ───────────
    {
        auto* header = new QHBoxLayout();
        header->setSpacing(8);

        auto* title_col = new QVBoxLayout();
        title_col->setSpacing(3);

        title_lbl_ = new QLabel("Open score CSV", this);
        title_lbl_->setFont(fonts::heading(20, 600));
        title_lbl_->setStyleSheet("color: #0b1640; margin: 0; padding: 0;");
        title_col->addWidget(title_lbl_);

        subtitle_lbl_ = new QLabel("Blue A score tables \xc2\xb7 accepts .csv", this);
        subtitle_lbl_->setFont(fonts::body(13, 400));
        subtitle_lbl_->setStyleSheet("color: #56658f; margin: 0; padding: 0;");
        title_col->addWidget(subtitle_lbl_);

        header->addLayout(title_col);
        header->addStretch();

        // Exact sleek vector close icon
        auto* close_btn = new QPushButton(this);
        close_btn->setObjectName("closeBtn");
        close_btn->setIcon(QIcon(render_svg(kCloseSvg, 14, 14)));
        close_btn->setIconSize(QSize(14, 14));
        close_btn->setCursor(Qt::PointingHandCursor);
        connect(close_btn, &QPushButton::clicked, this, &QDialog::reject);
        header->addWidget(close_btn, 0, Qt::AlignTop);

        root->addLayout(header);
    }

    // ── Mode Switcher Tabs ──────────────────────────────────────────────────
    {
        auto* tabs_row = new QHBoxLayout();
        tabs_row->setSpacing(8);

        tab_csv_btn_ = new QPushButton("Score CSV", this);
        tab_csv_btn_->setObjectName("tabBtnActive");
        tab_csv_btn_->setFont(fonts::sans(12, 600));
        tab_csv_btn_->setCursor(Qt::PointingHandCursor);
        connect(tab_csv_btn_, &QPushButton::clicked, this, [this]() {
            set_kind(SourceKind::Csv);
        });
        tabs_row->addWidget(tab_csv_btn_);

        tab_audio_btn_ = new QPushButton("Raw Audio (.wav \xc2\xb7 .flac)", this);
        tab_audio_btn_->setObjectName("tabBtnInactive");
        tab_audio_btn_->setFont(fonts::sans(12, 500));
        tab_audio_btn_->setCursor(Qt::PointingHandCursor);
        connect(tab_audio_btn_, &QPushButton::clicked, this, [this]() {
            set_kind(SourceKind::Audio);
        });
        tabs_row->addWidget(tab_audio_btn_);

        tabs_row->addStretch();
        root->addLayout(tabs_row);
    }

    // ── Drop Zone (min-height 220, exact matching SVG icon and fonts) ────────
    {
        drop_zone_frame_ = new QFrame(this);
        drop_zone_frame_->setObjectName("dropZone");
        drop_zone_frame_->setMinimumHeight(200);
        drop_zone_frame_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        auto* dz_layout = new QVBoxLayout(drop_zone_frame_);
        dz_layout->setContentsMargins(20, 24, 20, 24);
        dz_layout->setSpacing(8);
        dz_layout->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

        // Vector SVG download arrow matching web demo
        drop_icon_label_ = new QLabel(drop_zone_frame_);
        drop_icon_label_->setAlignment(Qt::AlignHCenter);
        drop_icon_label_->setPixmap(render_svg(kDropSvg, 36, 36));
        dz_layout->addWidget(drop_icon_label_, 0, Qt::AlignHCenter);

        // Clean text: "Drop files here"
        drop_title_label_ = new QLabel("Drop files here", drop_zone_frame_);
        drop_title_label_->setFont(fonts::sans(18, 600));
        drop_title_label_->setStyleSheet("color: #0b1640;");
        drop_title_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_title_label_, 0, Qt::AlignHCenter);

        drop_hint_label_ = new QLabel("Or choose files or a folder. Accepts .csv.", drop_zone_frame_);
        drop_hint_label_->setFont(fonts::mono(12, 450));
        drop_hint_label_->setStyleSheet("color: #56658f;");
        drop_hint_label_->setAlignment(Qt::AlignHCenter);
        dz_layout->addWidget(drop_hint_label_, 0, Qt::AlignHCenter);

        root->addWidget(drop_zone_frame_);
    }

    // ── Action Buttons (Right-aligned, height 42px, matching vector icons) ───
    {
        auto* btn_row = new QHBoxLayout();
        btn_row->setSpacing(8);
        btn_row->addStretch();

        auto* files_btn = new QPushButton("  Choose files", this);
        files_btn->setObjectName("actionBtn");
        files_btn->setIcon(QIcon(render_svg(kFilesSvg, 16, 16)));
        files_btn->setIconSize(QSize(16, 16));
        files_btn->setFont(fonts::sans(13, 500));
        files_btn->setCursor(Qt::PointingHandCursor);
        connect(files_btn, &QPushButton::clicked, this, &SourcePickerDialog::choose_files);
        btn_row->addWidget(files_btn);

        auto* folder_btn = new QPushButton("  Choose folder", this);
        folder_btn->setObjectName("actionBtn");
        folder_btn->setIcon(QIcon(render_svg(kFolderSvg, 16, 16)));
        folder_btn->setIconSize(QSize(16, 16));
        folder_btn->setFont(fonts::sans(13, 500));
        folder_btn->setCursor(Qt::PointingHandCursor);
        connect(folder_btn, &QPushButton::clicked, this, &SourcePickerDialog::choose_folder);
        btn_row->addWidget(folder_btn);

        root->addLayout(btn_row);
    }
}

void SourcePickerDialog::paintEvent(QPaintEvent* event) {
    QDialog::paintEvent(event);
    // Draw clean 1px border around the entire card
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(QPen(theme::kLine, 1.0));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 12.0, 12.0);
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

    // Automatically close dialog immediately upon selection
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
