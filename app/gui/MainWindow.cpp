#include "MainWindow.h"
#include "widgets/SourcePickerDialog.h"
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
#include <QPalette>
#include <QColor>
#include <QDir>
#include <QKeyEvent>
#include <QDirIterator>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrentRun>
#include <set>
#include "DisplayText.h"

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
    // 380px panel + room for the 8px scrollbar, so a long inspector never clips its right edge.
    detail_scroll->setFixedWidth(392);
    detail_scroll->setFrameShape(QFrame::NoFrame);
    detail_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    detail_scroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    detail_scroll->setWidgetResizable(true);
    detail_scroll->setAutoFillBackground(true);
    detail_scroll->viewport()->setAutoFillBackground(true);
    {
        QPalette pal = detail_scroll->palette();
        pal.setColor(QPalette::Window, QColor("#ffffff"));
        pal.setColor(QPalette::Base, QColor("#ffffff"));
        detail_scroll->setPalette(pal);
        detail_scroll->viewport()->setPalette(pal);
    }
    detail_scroll->setStyleSheet(
        "QScrollArea { background: #ffffff; border: none; }"
        "QScrollArea > QWidget { background: #ffffff; }"
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
    detail_panel_->setAutoFillBackground(true);
    {
        QPalette pal = detail_panel_->palette();
        pal.setColor(QPalette::Window, QColor("#ffffff"));
        pal.setColor(QPalette::Base, QColor("#ffffff"));
        detail_panel_->setPalette(pal);
    }
    detail_scroll->setWidget(detail_panel_);
    body_layout->addWidget(detail_scroll);
    detail_scroll_ = detail_scroll;
    detail_scroll_->hide();  // collapsed until the user clicks a row

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

        // TopBar triggers — open source picker modal
    auto open_source_picker = [this](gui::SourceKind initial_kind) {
        auto* dlg = new gui::SourcePickerDialog(initial_kind, this);
        connect(dlg, &gui::SourcePickerDialog::csv_file_chosen, this, [this](const QString& path) {
            open_file(path);
        });
        connect(dlg, &gui::SourcePickerDialog::audio_files_chosen, this, [this](const QStringList& paths) {
            ingest_paths(paths);
        });
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->exec();
    };

    connect(top_bar_, &gui::TopBar::open_requested, this, [open_source_picker]() {
        open_source_picker(gui::SourceKind::Csv);
    });
    connect(top_bar_, &gui::TopBar::import_audio_requested, this, [open_source_picker]() {
        open_source_picker(gui::SourceKind::Audio);
    });
    connect(top_bar_, &gui::TopBar::export_requested, this, &MainWindow::export_report);

    // Collapsible inspector: a mouse click on a row pops it open; its x button collapses it.
    connect(center_panel_, &gui::CenterPanel::row_clicked, this, [this](int) { show_inspector(); });
    connect(detail_panel_, &gui::DetailPanel::close_requested, this, &MainWindow::hide_inspector);

    // Compute mode (Auto / Hybrid / GPU / CPU), remembered across launches
    {
        QSettings settings;
        const QString mode = settings.value("compute_mode", "auto").toString();
        sidebar_->set_compute_mode(mode);
    }
    connect(sidebar_, &gui::Sidebar::compute_mode_changed, this, [this](const QString& mode) {
        QSettings().setValue("compute_mode", mode);
        engine_->set_device(mode);
        status_bar_->set_status(QString("Compute mode set to %1; it applies to the next scan.").arg(mode.toUpper()));
    });

    // DetailPanel label override
    connect(detail_panel_, &gui::DetailPanel::label_selected, this, &MainWindow::set_manual_label);

    connect(&state_, &AppState::status_message, status_bar_, &gui::StatusBar::set_status);

    engine_refresh_timer_ = new QTimer(this);
    engine_refresh_timer_->setSingleShot(true);
    engine_refresh_timer_->setInterval(250);  // at most ~4 view refreshes per second while streaming
    connect(engine_refresh_timer_, &QTimer::timeout, this, &MainWindow::flush_engine_view);

    engine_ = new EngineBridge(this);  // no worker process starts until audio is queued
    engine_->set_device(sidebar_->compute_mode());
    connect_engine();
}

void MainWindow::set_engine_bridge(EngineBridge* bridge) {
    if (engine_ && engine_ != bridge) engine_->deleteLater();
    engine_ = bridge;
    engine_->setParent(this);
    engine_->set_device(sidebar_->compute_mode());
    connect_engine();
}

void MainWindow::show_inspector() {
    if (detail_scroll_ && !detail_scroll_->isVisible()) detail_scroll_->show();
}

void MainWindow::hide_inspector() {
    if (detail_scroll_ && detail_scroll_->isVisible()) detail_scroll_->hide();
    detail_panel_->set_playing(false);
}

bool MainWindow::inspector_visible() const { return detail_scroll_ && !detail_scroll_->isHidden(); }

void MainWindow::set_unified_mode(bool unified) {
    const double thr = unified ? engine_->model_status().blue_a_threshold : gui::display::kDefaultBlueAThreshold;
    center_panel_->table()->set_unified_mode(unified);
    center_panel_->table()->set_threshold(thr);
    center_panel_->sync_mode();
    sidebar_->set_unified_model(unified, thr);
    detail_panel_->set_threshold(thr);
    status_bar_->set_version_text(unified ? QString::fromUtf8("v0.2 \xc2\xb7 WHALEMIX LIGHT")
                                          : QString::fromUtf8("v0.2 \xc2\xb7 BLUE A + LAYA"));
}

// Smoothed ETA: an exponential moving average of the files/second rate, sampled at most once a
// second, blended with the run's overall average so early bursts do not swing the estimate.
void MainWindow::update_progress(int done, int total, bool finished) {
    scan_total_ = total;
    double eta = -1.0;
    if (!finished && total > 0 && scan_clock_.isValid()) {
        const double t = scan_clock_.elapsed() / 1000.0;
        if (t - last_rate_t_ >= 1.0 && done > last_rate_done_) {
            const double inst = (done - last_rate_done_) / (t - last_rate_t_);
            smoothed_rate_ = smoothed_rate_ > 0.0 ? 0.7 * smoothed_rate_ + 0.3 * inst : inst;
            last_rate_t_ = t;
            last_rate_done_ = done;
        }
        const double overall = t > 0.0 ? done / t : 0.0;
        const double rate = smoothed_rate_ > 0.0 ? 0.5 * smoothed_rate_ + 0.5 * overall : overall;
        if (done > 0 && rate > 0.0 && t >= 2.0) eta = (total - done) / rate;
    }
    sidebar_->set_progress(done, total, eta, finished);
}

void MainWindow::connect_engine() {
    connect(engine_, &EngineBridge::backend_reported, this,
            [this](const QString&, const QString& active, int threads, const QString& note) {
                sidebar_->set_compute_backend(active, threads, note);
            });
    connect(engine_, &EngineBridge::model_unavailable, this, [this](const QString& reason) {
        status_bar_->set_status(QString("Model unavailable: %1. CSV viewing still works.").arg(reason));
    });
    connect(engine_, &EngineBridge::run_started, this, [this](const QString& run_dir, int total) {
        // Engine results replace the table; they are written under the run directory, not a CSV.
        state_.set_rows({}, QDir(run_dir).filePath("out/results.partial.jsonl"));
        set_unified_mode(true);
        hide_inspector();  // a new scan starts with the table at full width
        scan_clock_.start();
        smoothed_rate_ = 0.0;
        last_rate_t_ = 0.0;
        last_rate_done_ = 0;
        update_progress(0, total, false);
        top_bar_->set_file_name(QFileInfo(run_dir).fileName());
        sidebar_->set_source_info(QFileInfo(run_dir).fileName(), 0, 0);
        update_all_views();
        status_bar_->set_status(QString("Analyzing audio: 0 of %1 %2")
                                    .arg(total)
                                    .arg(gui::display::wav_file_unit(total)));
    });
    connect(engine_, &EngineBridge::rows_committed, this, [this](qint64) { append_engine_rows(); });
    connect(engine_, &EngineBridge::progress, this, [this](int done, int total, qint64 windows) {
        update_progress(done, total, false);
        status_bar_->set_status(QString("Analyzing audio: %1 of %2 %3, %4 %5, %6 results committed")
                                    .arg(done)
                                    .arg(total)
                                    .arg(gui::display::wav_file_unit(total))
                                    .arg(windows)
                                    .arg(gui::display::window_10s_unit(windows))
                                    .arg(engine_->committed_rows()));
    });
    connect(engine_, &EngineBridge::run_finished, this, [this](const QString& st, int, const QString& csv) {
        if (st != "rejected") flush_engine_view();  // show every committed row before the final status
        const QString kind = engine_->model_status().artifact_kind == "fixture-untrained" ? " (fixture model)" : "";
        if (st == "completed") {
            update_progress(scan_total_, scan_total_, true);
            status_bar_->set_status(QString("Analysis complete%1: %2 results in %3")
                                        .arg(kind).arg(engine_->committed_rows()).arg(csv));
        } else if (st == "canceled") {
            status_bar_->set_status(QString("Analysis canceled: %1 committed results kept as partial output in %2")
                                        .arg(engine_->committed_rows()).arg(csv));
        } else if (st != "rejected") {
            status_bar_->set_status(QString("Analysis %1; committed results kept in %2").arg(st, engine_->run_dir()));
        }
    });
}

void MainWindow::append_engine_rows() {
    const QList<QJsonObject> fresh = engine_->take_new_rows();
    if (fresh.isEmpty()) return;
    const bool first = state_.rows().empty();
    auto& rows = state_.mutable_rows();
    for (const QJsonObject& obj : fresh) rows.push_back(row_from_engine_json(obj));
    if (first) {
        state_.set_rows(std::move(rows), state_.source_path());  // picks the first call, emits signals
        update_all_views();
        return;
    }
    // Rows are kept in state immediately; the views catch up on a short coalescing timer so a run
    // of thousands of files never rebuilds the whole table per committed batch.
    if (!engine_refresh_timer_->isActive()) engine_refresh_timer_->start();
}

void MainWindow::flush_engine_view() {
    if (engine_refresh_timer_) engine_refresh_timer_->stop();
    const auto& rows = state_.rows();
    const size_t from = center_panel_->shown_rows();
    if (from >= rows.size()) return;
    for (size_t i = from; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (r.mix_label == "call") engine_counts_[0]++;
        else if (r.mix_label == "not-call") engine_counts_[1]++;
        else if (r.mix_label == "review") engine_counts_[2]++;
        if (!r.sample_id.empty()) engine_sounds_.insert(r.sample_id);
    }
    const int total = static_cast<int>(rows.size());
    sidebar_->set_counts(engine_counts_[0], engine_counts_[1], engine_counts_[2], total);
    sidebar_->set_source_info(QFileInfo(engine_->run_dir()).fileName(), static_cast<int>(engine_sounds_.size()), total);
    center_panel_->append_rows(rows, from, !state_.source_path().isEmpty(),
                               static_cast<int>(engine_sounds_.size()));
}

void MainWindow::ingest_paths(const QStringList& paths) {
    QStringList csvs;
    QStringList audio_or_dirs;
    int skipped = 0;
    for (const QString& p : paths) {
        QFileInfo fi(p);
        if (fi.isDir() || EngineBridge::is_audio_path(p)) {
            audio_or_dirs << p;
        } else if (fi.suffix().compare("csv", Qt::CaseInsensitive) == 0) {
            csvs << p;
        } else {
            ++skipped;
        }
    }
    if (!csvs.isEmpty()) open_file(csvs.first());
    if (audio_or_dirs.isEmpty()) {
        if (skipped > 0 && csvs.isEmpty()) status_bar_->set_status(QString("Skipped %1 unsupported files").arg(skipped));
        return;
    }
    const ModelStatus st = engine_->model_status();
    if (!st.available) {
        status_bar_->set_status(QString("Model unavailable: %1. CSV viewing still works.").arg(st.reason));
        return;
    }
    status_bar_->set_status("Discovering audio files…");
    auto* watcher = new QFutureWatcher<QStringList>(this);
    connect(watcher, &QFutureWatcher<QStringList>::finished, this, [this, watcher, skipped]() {
        start_audio_analysis(watcher->result(), skipped);
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([audio_or_dirs]() {
        QStringList files;
        for (const QString& p : audio_or_dirs) {
            if (!QFileInfo(p).isDir()) {
                files << p;
                continue;
            }
            QDirIterator it(p, QDir::Files | QDir::Readable, QDirIterator::Subdirectories);
            while (it.hasNext()) {
                const QString f = it.next();
                if (EngineBridge::is_audio_path(f)) files << f;
            }
        }
        files.sort();
        return files;
    }));
}

void MainWindow::start_audio_analysis(const QStringList& audio_files, int skipped) {
    if (audio_files.isEmpty()) {
        status_bar_->set_status("No supported audio files found");
        return;
    }
    if (engine_->running()) {
        status_bar_->set_status("An analysis is already running; cancel it before starting another");
        return;
    }
    engine_->analyze(audio_files);
    if (skipped > 0) status_bar_->set_status(status_bar_->status() + QString(" (%1 unsupported files skipped)").arg(skipped));
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
        set_unified_mode(!state_.rows().empty() && gui::display::is_unified_row(state_.rows().front()));
        sidebar_->set_progress(0, 0, -1.0, false);
        hide_inspector();

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
    engine_counts_[0] = call;
    engine_counts_[1] = not_call;
    engine_counts_[2] = review;
    engine_sounds_ = sounds;

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

    const std::string old_label = r.mix_label;
    whalemix::set_label(r, label.toStdString());
    state_.set_unsaved_overrides(true);

    refresh_row(idx, old_label);

    QString win_name = QString::fromStdString(r.candidate_id.empty() ? r.sample_id : r.candidate_id);
    status_bar_->set_status(QString("%1 set to %2").arg(win_name).arg(label));
}

// A single row changed label: update counts, that table row and the detail panel without
// rebuilding every row (a 10,000-file run has ~335k rows).
void MainWindow::refresh_row(int idx, const std::string& old_label) {
    if (center_panel_->shown_rows() != state_.rows().size()) {
        update_all_views();
        return;
    }
    const auto& r = state_.rows()[static_cast<size_t>(idx)];
    auto bump = [this](const std::string& label, int d) {
        if (label == "call") engine_counts_[0] += d;
        else if (label == "not-call") engine_counts_[1] += d;
        else if (label == "review") engine_counts_[2] += d;
    };
    bump(old_label, -1);
    bump(r.mix_label, +1);
    sidebar_->set_counts(engine_counts_[0], engine_counts_[1], engine_counts_[2], static_cast<int>(state_.rows().size()));
    bool has_manual = r.mix_reason == "manual";
    for (const auto& e : undo_stack_) {
        if (has_manual) break;
        const auto& u = state_.rows()[static_cast<size_t>(e.row_index)];
        has_manual = u.mix_reason == "manual";
    }
    sidebar_->set_has_manual(has_manual);
    center_panel_->update_row(static_cast<size_t>(idx), r, old_label);
    if (idx == state_.selected_index()) {
        detail_panel_->set_row(&r, QFileInfo(state_.source_path()).absolutePath());
    }
}

void MainWindow::undo_override() {
    if (undo_stack_.empty()) return;

    UndoEntry entry = undo_stack_.back();
    undo_stack_.pop_back();

    if (entry.row_index >= 0 && entry.row_index < static_cast<int>(state_.rows().size())) {
        auto& r = state_.mutable_rows()[entry.row_index];
        const std::string old_label = r.mix_label;
        r.mix_label = entry.prev_label;
        r.mix_reason = entry.prev_reason;
        refresh_row(entry.row_index, old_label);

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
    if (!event->mimeData()->hasUrls()) return;
    for (const QUrl& url : event->mimeData()->urls()) {
        const QString path = url.toLocalFile();
        QFileInfo fi(path);
        if (fi.isDir() || EngineBridge::is_audio_path(path) ||
            fi.suffix().compare("csv", Qt::CaseInsensitive) == 0) {
            event->acceptProposedAction();
            return;
        }
    }
}

void MainWindow::dropEvent(QDropEvent* event) {
    QStringList paths;
    for (const QUrl& url : event->mimeData()->urls()) {
        if (url.isLocalFile()) paths << url.toLocalFile();
    }
    if (paths.isEmpty()) return;
    event->acceptProposedAction();
    ingest_paths(paths);
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

    if (event->key() == Qt::Key_Escape) {
        hide_inspector();
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
    if (event->key() == Qt::Key_R && !center_panel_->table()->unified_mode()) {
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
