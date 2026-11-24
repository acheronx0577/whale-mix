#include "MainWindow.h"
#include "Theme.h"
#include "Fonts.h"
#include "whalemix/table.h"
#include "whalemix/exporter.h"
#include <QPainter>
#include <QIcon>
#include <QScrollArea>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QAbstractButton>
#include <QSettings>
#include <QFileInfo>
#include <QDir>
#include <QKeyEvent>
#include <set>

#ifdef Q_OS_WIN
#include <windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#endif

namespace whalemix {

namespace {

QString format_home_path(const QString& full_path) {
    QString home = QDir::homePath();
    QString native = QDir::toNativeSeparators(full_path);
    QString native_home = QDir::toNativeSeparators(home);
    if (native.startsWith(native_home)) {
        return "~" + native.mid(native_home.length());
    }
    return native;
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint | Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint | Qt::WindowSystemMenuHint);
    setMouseTracking(true);
    resize(1440, 900);
    setMinimumSize(1200, 760);
    setWindowTitle("Whale Mix");
    setWindowIcon(QIcon(":/icons/whale-logo-native-53.png"));
    setAcceptDrops(true);

    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    top_bar_ = new gui::TopBar(central);
    layout->addWidget(top_bar_);

    auto* body = new QWidget(central);
    body->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* body_layout = new QHBoxLayout(body);
    body_layout->setContentsMargins(0, 0, 0, 0);
    body_layout->setSpacing(0);

    sidebar_ = new gui::Sidebar(body);
    body_layout->addWidget(sidebar_);

    center_panel_ = new gui::CenterPanel(body);
    body_layout->addWidget(center_panel_, 1);

    auto* detail_scroll = new QScrollArea(body);
    detail_scroll->setFixedWidth(380);
    detail_scroll->setFrameShape(QFrame::NoFrame);
    detail_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    detail_scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    detail_scroll->setWidgetResizable(true);
    detail_scroll->setStyleSheet(
        "QScrollArea { background: white; border: none; }"
        "QScrollBar:vertical {"
        "  width: 8px;"
        "  background: transparent;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #d9e4f7;"
        "  min-height: 24px;"
        "  border-radius: 4px;"
        "}"
        "QScrollBar::handle:vertical:hover {"
        "  background: #2563eb;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
        "  height: 0px;"
        "}"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
        "  background: transparent;"
        "}"
    );

    detail_panel_ = new gui::DetailPanel(detail_scroll);
    detail_scroll->setWidget(detail_panel_);
    body_layout->addWidget(detail_scroll);

    layout->addWidget(body);

    status_bar_ = new gui::StatusBar(central);
    layout->addWidget(status_bar_);

    // Sync pile filters between Sidebar and CenterPanel
    connect(sidebar_, &gui::Sidebar::pile_filter_changed, center_panel_, &gui::CenterPanel::set_active_pile);
    connect(center_panel_, &gui::CenterPanel::pile_filter_changed, sidebar_, &gui::Sidebar::set_active_pile);

    // Sync selection from CenterPanel -> AppState
    connect(center_panel_, &gui::CenterPanel::selection_changed, &state_, &AppState::set_selected_index);

    // When selection changes in AppState -> update CenterPanel table and DetailPanel
    connect(&state_, &AppState::selection_changed, this, [this](int idx) {
        if (center_panel_->table()->selected_model_index() != idx) {
            center_panel_->table()->select_model_index(idx);
        }
        if (idx >= 0 && idx < static_cast<int>(state_.rows().size())) {
            QFileInfo fi(state_.source_path());
            detail_panel_->set_row(&state_.rows()[idx], fi.absolutePath());
        } else {
            detail_panel_->set_row(nullptr, "");
        }
    });

    // TopBar triggers
    connect(top_bar_, &gui::TopBar::open_requested, this, [this]() {
        QSettings settings;
        QString last_dir = settings.value("last_open_dir", QDir::homePath()).toString();
        QString file = QFileDialog::getOpenFileName(this, "Open Score CSV", last_dir, "Blue A score CSV (*.csv)");
        if (!file.isEmpty()) {
            settings.setValue("last_open_dir", QFileInfo(file).absolutePath());
            open_file(file);
        }
    });
    connect(top_bar_, &gui::TopBar::import_audio_requested, this, [this]() {
        QSettings settings;
        QString last_dir = settings.value("last_audio_dir", QDir::homePath()).toString();
        QString file = QFileDialog::getOpenFileName(this, "Import Raw Audio", last_dir, "Audio Files (*.wav *.flac *.ogg)");
        if (!file.isEmpty()) {
            settings.setValue("last_audio_dir", QFileInfo(file).absolutePath());
            status_bar_->set_status(QString("Loaded raw audio %1 — generate spectrogram or score").arg(QFileInfo(file).fileName()));
        }
    });
    connect(top_bar_, &gui::TopBar::export_requested, this, &MainWindow::export_report);

    // DetailPanel label override
    connect(detail_panel_, &gui::DetailPanel::label_selected, this, &MainWindow::set_manual_label);

    connect(&state_, &AppState::status_message, status_bar_, &gui::StatusBar::set_status);
}

void MainWindow::open_file(const QString& path) {
    // Unsaved changes guard
    if (state_.has_unsaved_overrides()) {
        int manual_count = 0;
        for (const auto& r : state_.rows()) {
            if (r.mix_reason == "manual") manual_count++;
        }
        QString msg = QString("%1 labels were set by hand and are not exported yet.").arg(manual_count);
        QMessageBox box(QMessageBox::Question, "Unsaved manual labels", msg, QMessageBox::NoButton, this);
        auto* exp_btn = box.addButton("Export…", QMessageBox::AcceptRole);
        auto* disc_btn = box.addButton("Discard", QMessageBox::DestructiveRole);
        box.addButton("Cancel", QMessageBox::RejectRole);
        box.exec();

        if (box.clickedButton() == static_cast<QAbstractButton*>(exp_btn)) {
            export_report();
            if (state_.has_unsaved_overrides()) return; // User cancelled export
        } else if (box.clickedButton() != static_cast<QAbstractButton*>(disc_btn)) {
            return; // Cancelled
        }
    }

    try {
        LoadResult res = rows_from_blue_a(path.toStdString());
        QFileInfo fi(path);

        setWindowTitle(QString("Whale Mix — %1").arg(fi.fileName()));
        top_bar_->set_file_name(fi.fileName());
        undo_stack_.clear();

        state_.set_rows(std::move(res.rows), path);

        // Compute unique sounds count
        std::set<std::string> sounds;
        for (const auto& r : state_.rows()) {
            if (!r.sample_id.empty()) sounds.insert(r.sample_id);
        }
        int num_sounds = static_cast<int>(sounds.size());

        sidebar_->set_source_info(fi.fileName(), num_sounds, static_cast<int>(state_.rows().size()));
        center_panel_->set_rows(state_.rows(), true, num_sounds);
        center_panel_->set_active_pile("all");
        sidebar_->set_active_pile("all");

        center_panel_->table()->select_first_call_or_first();

        // Format status string
        QString status = QString("%1 pictures from %2").arg(state_.rows().size()).arg(format_home_path(path));
        if (res.skipped_not_scored > 0) {
            status += QString(" (%1 skipped: not scored)").arg(res.skipped_not_scored);
        }
        if (res.skipped_bad_score > 0) {
            status += QString(" (%1 skipped: no bat or baf)").arg(res.skipped_bad_score);
        }
        status_bar_->set_status(status);

        update_all_views();

    } catch (const std::exception& e) {
        QString err = e.what();
        status_bar_->set_status(err);
        QMessageBox::critical(this, "Could not open score CSV", err);
    }
}

void MainWindow::update_all_views() {
    int call = 0;
    int not_call = 0;
    int review = 0;
    bool has_manual = false;
    std::set<std::string> sounds;

    for (const auto& r : state_.rows()) {
        if (r.mix_label == "call") call++;
        else if (r.mix_label == "not-call") not_call++;
        else if (r.mix_label == "review") review++;
        if (r.mix_reason == "manual") has_manual = true;
        if (!r.sample_id.empty()) sounds.insert(r.sample_id);
    }
    int total = static_cast<int>(state_.rows().size());
    int num_sounds = static_cast<int>(sounds.size());

    sidebar_->set_counts(call, not_call, review, total);
    sidebar_->set_has_manual(has_manual);

    center_panel_->set_rows(state_.rows(), !state_.source_path().isEmpty(), num_sounds);

    int idx = state_.selected_index();
    if (idx >= 0 && idx < total) {
        QFileInfo fi(state_.source_path());
        detail_panel_->set_row(&state_.rows()[idx], fi.absolutePath());
    } else {
        detail_panel_->set_row(nullptr, "");
    }
}

void MainWindow::set_manual_label(const QString& label) {
    int idx = state_.selected_index();
    if (idx < 0 || idx >= static_cast<int>(state_.rows().size())) return;

    auto& r = state_.mutable_rows()[idx];
    undo_stack_.push_back(UndoEntry{idx, r.mix_label, r.mix_reason});

    whalemix::set_label(r, label.toStdString());
    state_.set_unsaved_overrides(true);

    update_all_views();

    QString win_name = QString::fromStdString(r.candidate_id.empty() ? r.sample_id : r.candidate_id);
    status_bar_->set_status(QString("%1 set to %2").arg(win_name).arg(label));
}

void MainWindow::undo_override() {
    if (undo_stack_.empty()) return;

    UndoEntry entry = undo_stack_.back();
    undo_stack_.pop_back();

    if (entry.row_index >= 0 && entry.row_index < static_cast<int>(state_.rows().size())) {
        auto& r = state_.mutable_rows()[entry.row_index];
        r.mix_label = entry.prev_label;
        r.mix_reason = entry.prev_reason;
        update_all_views();

        QString win_name = QString::fromStdString(r.candidate_id.empty() ? r.sample_id : r.candidate_id);
        status_bar_->set_status(QString("Restored %1").arg(win_name));
    }
}

void MainWindow::export_report() {
    if (state_.rows().empty()) {
        status_bar_->set_status("Open a score CSV first.");
        return;
    }

    QSettings settings;
    QString default_dir = settings.value("last_export_dir", "").toString();
    if (default_dir.isEmpty() && !state_.source_path().isEmpty()) {
        default_dir = QFileInfo(state_.source_path()).absolutePath();
    }
    if (default_dir.isEmpty()) {
        default_dir = QDir::homePath();
    }

    QString out_dir = QFileDialog::getExistingDirectory(this, "Select Export Folder", default_dir);
    if (out_dir.isEmpty()) return;

    settings.setValue("last_export_dir", out_dir);

    // Confirm replace if decisions.csv already exists
    QDir d(out_dir);
    if (d.exists("decisions.csv")) {
        auto res = QMessageBox::question(
            this, "Confirm Replace",
            QString("Replace the export in %1?").arg(out_dir),
            QMessageBox::Yes | QMessageBox::No
        );
        if (res != QMessageBox::Yes) return;
    }

    try {
        whalemix::export_rows(state_.rows(), out_dir.toStdString());
        state_.set_unsaved_overrides(false);

        int call = 0, not_call = 0, review = 0;
        for (const auto& r : state_.rows()) {
            if (r.mix_label == "call") call++;
            else if (r.mix_label == "not-call") not_call++;
            else if (r.mix_label == "review") review++;
        }
        status_bar_->set_status(
            QString("Wrote %1 pictures to %2 (call %3, not-call %4, review %5)")
                .arg(state_.rows().size())
                .arg(out_dir)
                .arg(call).arg(not_call).arg(review)
        );
    } catch (const std::exception& e) {
        QString err = e.what();
        status_bar_->set_status(err);
        QMessageBox::critical(this, "Export Error", err);
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        const auto& urls = event->mimeData()->urls();
        if (!urls.isEmpty() && urls.first().toLocalFile().endsWith(".csv", Qt::CaseInsensitive)) {
            event->acceptProposedAction();
        }
    }
}

void MainWindow::dropEvent(QDropEvent* event) {
    const auto& urls = event->mimeData()->urls();
    if (!urls.isEmpty()) {
        QString path = urls.first().toLocalFile();
        if (path.endsWith(".csv", Qt::CaseInsensitive)) {
            event->acceptProposedAction();
            open_file(path);
        }
    }
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (state_.has_unsaved_overrides()) {
        int manual_count = 0;
        for (const auto& r : state_.rows()) {
            if (r.mix_reason == "manual") manual_count++;
        }
        QString msg = QString("%1 labels were set by hand and are not exported yet.").arg(manual_count);
        QMessageBox box(QMessageBox::Question, "Unsaved manual labels", msg, QMessageBox::NoButton, this);
        auto* exp_btn = box.addButton("Export…", QMessageBox::AcceptRole);
        auto* disc_btn = box.addButton("Discard", QMessageBox::DestructiveRole);
        box.addButton("Cancel", QMessageBox::RejectRole);
        box.exec();

        if (box.clickedButton() == static_cast<QAbstractButton*>(exp_btn)) {
            export_report();
            if (state_.has_unsaved_overrides()) {
                event->ignore();
                return;
            }
            event->accept();
        } else if (box.clickedButton() == static_cast<QAbstractButton*>(disc_btn)) {
            event->accept();
        } else {
            event->ignore();
            return;
        }
    } else {
        event->accept();
    }
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    // If search box input is focused, don't hijack letters
    if (center_panel_ && center_panel_->search_box()->hasFocus()) {
        QMainWindow::keyPressEvent(event);
        return;
    }

    bool ctrl = (event->modifiers() & Qt::ControlModifier) || (event->modifiers() & Qt::MetaModifier);

    if (ctrl && event->key() == Qt::Key_O) {
        top_bar_->open_requested();
        return;
    }
    if ((ctrl && event->key() == Qt::Key_E) || (!ctrl && event->key() == Qt::Key_E)) {
        export_report();
        return;
    }
    if (ctrl && event->key() == Qt::Key_F) {
        center_panel_->focus_search();
        return;
    }
    if (ctrl && event->key() == Qt::Key_Z) {
        undo_override();
        return;
    }

    if (event->key() == Qt::Key_Space) {
        if (detail_panel_) {
            detail_panel_->toggle_play();
        }
        return;
    }
    if (event->key() == Qt::Key_A && !ctrl) {
        if (detail_panel_) {
            detail_panel_->cycle_gain();
        }
        return;
    }
    if (event->key() == Qt::Key_J || event->key() == Qt::Key_Down) {
        center_panel_->table()->select_next();
        return;
    }
    if (event->key() == Qt::Key_K || event->key() == Qt::Key_Up) {
        center_panel_->table()->select_previous();
        return;
    }
    if (event->key() == Qt::Key_PageDown) {
        center_panel_->table()->page_down();
        return;
    }
    if (event->key() == Qt::Key_PageUp) {
        center_panel_->table()->page_up();
        return;
    }
    if (event->key() == Qt::Key_Home) {
        center_panel_->table()->select_first();
        return;
    }
    if (event->key() == Qt::Key_End) {
        center_panel_->table()->select_last();
        return;
    }
    if (event->key() == Qt::Key_C) {
        set_manual_label("call");
        return;
    }
    if (event->key() == Qt::Key_N) {
        set_manual_label("not-call");
        return;
    }
    if (event->key() == Qt::Key_R) {
        set_manual_label("review");
        return;
    }

    QMainWindow::keyPressEvent(event);
}

void MainWindow::showEvent(QShowEvent* event) {
    QMainWindow::showEvent(event);
#ifdef Q_OS_WIN
    HWND hwnd = reinterpret_cast<HWND>(winId());
    LONG_PTR style = GetWindowLongPtr(hwnd, GWL_STYLE);
    style |= (WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU);
    SetWindowLongPtr(hwnd, GWL_STYLE, style);
    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    MARGINS margins = {1, 1, 1, 1};
    DwmExtendFrameIntoClientArea(hwnd, &margins);
#endif
}

void MainWindow::paintEvent(QPaintEvent* event) {
    QMainWindow::paintEvent(event);
    if (!isMaximized()) {
        QPainter p(this);
        p.setPen(QPen(theme::kLine, 1.0));
        p.drawRect(0, 0, width() - 1, height() - 1);
    }
}

bool MainWindow::nativeEvent(const QByteArray& eventType, void* message, qintptr* result) {
#ifdef Q_OS_WIN
    if (eventType == "windows_generic_MSG") {
        MSG* msg = static_cast<MSG*>(message);
        if (msg->message == WM_SYSCOMMAND) {
            WPARAM wp = (msg->wParam & 0xFFF0);
            if (wp == SC_MINIMIZE) {
                showMinimized();
                *result = 0;
                return true;
            } else if (wp == SC_RESTORE) {
                showNormal();
                *result = 0;
                return true;
            }
        }
        if (msg->message == WM_NCHITTEST && !isMaximized()) {
            POINT pt = { GET_X_LPARAM(msg->lParam), GET_Y_LPARAM(msg->lParam) };
            RECT winRect;
            GetWindowRect(msg->hwnd, &winRect);
            int border = 8;

            bool left = pt.x >= winRect.left && pt.x < winRect.left + border;
            bool right = pt.x <= winRect.right && pt.x > winRect.right - border;
            bool top = pt.y >= winRect.top && pt.y < winRect.top + border;
            bool bottom = pt.y <= winRect.bottom && pt.y > winRect.bottom - border;

            if (top && left) { *result = HTTOPLEFT; return true; }
            if (top && right) { *result = HTTOPRIGHT; return true; }
            if (bottom && left) { *result = HTBOTTOMLEFT; return true; }
            if (bottom && right) { *result = HTBOTTOMRIGHT; return true; }
            if (left) { *result = HTLEFT; return true; }
            if (right) { *result = HTRIGHT; return true; }
            if (top) { *result = HTTOP; return true; }
            if (bottom) { *result = HTBOTTOM; return true; }
        }
    }
#endif
    return QMainWindow::nativeEvent(eventType, message, result);
}

} // namespace whalemix
