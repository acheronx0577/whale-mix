#include <gtest/gtest.h>
#include <QApplication>
#include <cmath>
#include <QDataStream>
#include <QTemporaryDir>
#include <QFile>
#include <QImage>
#include <QPixmap>
#include "Fonts.h"
#include "MainWindow.h"
#include "widgets/Chip.h"
#include "widgets/Kbd.h"
#include "widgets/PillButton.h"
#include "widgets/ScoreTable.h"
#include "widgets/SpectrogramView.h"
#include "whalemix/table.h"
#include "whalemix/csv.h"
#include "DisplayText.h"
#include <QTest>
#include <QSignalSpy>
#include <QScrollArea>
#include "widgets/DetailPanel.h"
#include "widgets/Sidebar.h"
#include <QFontMetrics>

TEST(GuiSmokeTest, WindowCreation) {
    whalemix::MainWindow window;
    EXPECT_EQ(window.windowTitle().toStdString(), "Whale Mix");
    ASSERT_NE(window.top_bar(), nullptr);
    ASSERT_NE(window.status_bar(), nullptr);
    ASSERT_NE(window.sidebar(), nullptr);
    ASSERT_NE(window.center_panel(), nullptr);
    EXPECT_EQ(window.top_bar()->sizeHint().height(), 72);
    EXPECT_EQ(window.status_bar()->sizeHint().height(), 36);
    EXPECT_EQ(window.sidebar()->sizeHint().width(), 240);
    EXPECT_EQ(window.status_bar()->status().toStdString(), "Open a Blue A score CSV.");
}

static std::filesystem::path find_sample_csv() {
    namespace fs = std::filesystem;
#ifdef WHALEMIX_SOURCE_DIR
    fs::path src_dir = WHALEMIX_SOURCE_DIR;
    if (fs::exists(src_dir / "samples" / "blue_a_scores.csv")) {
        return src_dir / "samples" / "blue_a_scores.csv";
    }
#endif
    std::vector<fs::path> candidates = {
        fs::current_path() / "samples" / "blue_a_scores.csv",
        fs::current_path().parent_path() / "samples" / "blue_a_scores.csv",
        fs::current_path().parent_path().parent_path() / "samples" / "blue_a_scores.csv",
        fs::current_path().parent_path().parent_path().parent_path() / "samples" / "blue_a_scores.csv",
        "D:/Code/Projects/Marine-Science/MBARI-Portfolio/Blue-Whale-Call-Detection-Under-Ocean-Noise/samples/blue_a_scores.csv"
    };
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c;
    }
    return "";
}

TEST(GuiSmokeTest, FullWorkflowWithOverridesAndUndo) {
    whalemix::MainWindow window;
    auto sample_path = find_sample_csv();
    ASSERT_FALSE(sample_path.empty());

    window.open_file(QString::fromStdString(sample_path.string()));

    auto* table = window.center_panel()->table();
    EXPECT_EQ(table->visible_count(), 69);

    // Initial selection is sample-a-004
    EXPECT_EQ(table->selected_model_index(), 3);
    EXPECT_EQ(window.state()->rows()[table->selected_model_index()].candidate_id, "sample-a-004");
    EXPECT_EQ(window.state()->rows()[table->selected_model_index()].mix_label, "call");

    // 'J' key -> next visible row (sample-a-005)
    table->select_next();
    EXPECT_EQ(table->selected_model_index(), 4);
    EXPECT_EQ(window.state()->rows()[table->selected_model_index()].candidate_id, "sample-a-005");

    // Back to sample-a-004
    table->select_previous();
    EXPECT_EQ(table->selected_model_index(), 3);

    // Set 'R' (review) on sample-a-004
    window.set_manual_label("review");
    EXPECT_EQ(window.state()->rows()[3].mix_label, "review");
    EXPECT_EQ(window.state()->rows()[3].mix_reason, "manual");

    // Check counts: call should be 0, review should be 2, not-call 67
    int call = 0, not_call = 0, review = 0;
    for (const auto& r : window.state()->rows()) {
        if (r.mix_label == "call") call++;
        else if (r.mix_label == "not-call") not_call++;
        else if (r.mix_label == "review") review++;
    }
    EXPECT_EQ(call, 0);
    EXPECT_EQ(review, 2);
    EXPECT_EQ(not_call, 67);

    // Undo (Ctrl Z) restores
    window.undo_override();
    EXPECT_EQ(window.state()->rows()[3].mix_label, "call");
    EXPECT_EQ(window.state()->rows()[3].mix_reason, "blue-a");
}

TEST(GuiSmokeTest, SharedPieces) {
    EXPECT_TRUE(QFile::exists(":/icons/whale-logo-native-53.png"));
    EXPECT_TRUE(QFile::exists(":/icons/folder-open.svg"));
    QPixmap px(":/icons/whale-logo-native-53.png");
    EXPECT_FALSE(px.isNull());
    QImage img(":/icons/whale-logo-native-53.png");
    EXPECT_FALSE(img.isNull());

    // Chip
    whalemix::gui::Chip chip("call");
    EXPECT_EQ(chip.label().toStdString(), "call");
    EXPECT_EQ(chip.sizeHint().height(), 22);

    // Kbd
    whalemix::gui::Kbd kbd("Ctrl O");
    EXPECT_EQ(kbd.key().toStdString(), "Ctrl O");
    EXPECT_GE(kbd.sizeHint().height(), 20);

    // PillButton
    whalemix::gui::PillButton pill("All", 69);
    EXPECT_EQ(pill.name().toStdString(), "All");
    EXPECT_EQ(pill.count(), 69);
    EXPECT_EQ(pill.sizeHint().height(), 32);
}

TEST(GuiSmokeTest, TypographyDesignSystem) {
    whalemix::fonts::init();

    // 1. Google Sans Flex (strictly only Google Sans Flex, zero other fonts)
    EXPECT_EQ(whalemix::fonts::sans_family().toStdString(), "Google Sans Flex");
    const auto& sans_fams = whalemix::fonts::sans_families();
    EXPECT_EQ(sans_fams.size(), 1);
    EXPECT_EQ(sans_fams.first().toStdString(), "Google Sans Flex");

    // Headings (H1–H6): font-weight: 450, letter-spacing: -2% (-0.02)
    QFont h1 = whalemix::fonts::heading(28);
    EXPECT_EQ(h1.pixelSize(), 28);
    EXPECT_EQ(static_cast<int>(h1.weight()), 450);
    EXPECT_NEAR(h1.letterSpacing(), -0.02 * 28.0, 0.02);
    EXPECT_EQ(h1.family().toStdString(), "Google Sans Flex");

    // Body (p): font-weight: 400, letter-spacing: -2% (-0.02)
    QFont p = whalemix::fonts::body(14);
    EXPECT_EQ(p.pixelSize(), 14);
    EXPECT_EQ(static_cast<int>(p.weight()), 400);
    EXPECT_NEAR(p.letterSpacing(), -0.02 * 14.0, 0.02);
    EXPECT_EQ(p.family().toStdString(), "Google Sans Flex");

    // 2. Google Sans Code (strictly only Google Sans Code, zero other fonts)
    EXPECT_EQ(whalemix::fonts::mono_family().toStdString(), "Google Sans Code");
    const auto& mono_fams = whalemix::fonts::mono_families();
    EXPECT_EQ(mono_fams.size(), 1);
    EXPECT_EQ(mono_fams.first().toStdString(), "Google Sans Code");

    // Monospace settings: font-weight: 450
    QFont m = whalemix::fonts::mono(13);
    EXPECT_EQ(m.pixelSize(), 13);
    EXPECT_EQ(static_cast<int>(m.weight()), 450);
    EXPECT_EQ(m.family().toStdString(), "Google Sans Code");
}

TEST(GuiSmokeTest, DetailPanelPlaybackAndAmplify) {
    whalemix::MainWindow window;
    auto sample_path = find_sample_csv();
    ASSERT_FALSE(sample_path.empty());
    window.open_file(QString::fromStdString(sample_path.string()));

    auto* detail = window.detail_panel();
    ASSERT_NE(detail, nullptr);

    // Initial playback and gain
    EXPECT_FALSE(detail->is_playing());
    EXPECT_EQ(detail->gain(), 1);
    EXPECT_DOUBLE_EQ(detail->playhead(), 0.0);

    // Toggle play
    detail->toggle_play();
    EXPECT_TRUE(detail->is_playing());

    // Cycle gain: 1 -> 2 -> 4 -> 8 -> 1
    detail->cycle_gain();
    EXPECT_EQ(detail->gain(), 2);
    detail->cycle_gain();
    EXPECT_EQ(detail->gain(), 4);
    detail->cycle_gain();
    EXPECT_EQ(detail->gain(), 8);
    detail->cycle_gain();
    EXPECT_EQ(detail->gain(), 1);

    // Set playhead
    detail->set_playhead(0.5);
    EXPECT_DOUBLE_EQ(detail->playhead(), 0.5);

    detail->toggle_play();
    EXPECT_FALSE(detail->is_playing());

    // Render snapshot offscreen to ensure no painting crash and save artifact
    window.resize(1440, 800);
    QPixmap shot = window.grab();
    EXPECT_FALSE(shot.isNull());
    EXPECT_EQ(shot.width(), 1440);
    EXPECT_EQ(shot.height(), 800);
    shot.save("C:/Users/IamTh/.gemini/antigravity/brain/f0c1155d-c01c-4413-86a8-5249f9fad1a1/cpp_app_parity_verified.png");
}

TEST(GuiSmokeTest, EngineRowSpectrogramRendersFromWav) {
    // A 2 kHz WAV with an 80 Hz tone: the rendered 0-250 Hz window must be darkest near 80 Hz.
    QTemporaryDir tmp;
    const QString path = tmp.filePath("tone.wav");
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        const quint32 rate = 2000, n = rate * 12;
        QDataStream s(&f);
        s.setByteOrder(QDataStream::LittleEndian);
        s.writeRawData("RIFF", 4);
        s << quint32(36 + n * 2);
        s.writeRawData("WAVEfmt ", 8);
        s << quint32(16) << quint16(1) << quint16(1) << rate << quint32(rate * 2) << quint16(2) << quint16(16);
        s.writeRawData("data", 4);
        s << quint32(n * 2);
        for (quint32 i = 0; i < n; ++i) s << qint16(std::lround(12000.0 * std::sin(2 * M_PI * 80.0 * i / rate)));
    }
    const QImage img = whalemix::gui::SpectrogramView::render_wav_window(path, 1.0, 11.0, 0);
    ASSERT_FALSE(img.isNull());
    int darkest_row = -1, darkest = 256;
    for (int y = 0; y < img.height(); ++y) {
        const int v = qRed(img.pixel(img.width() / 2, y));
        if (v < darkest) { darkest = v; darkest_row = y; }
    }
    const double hz = 250.0 * (img.height() - 1 - darkest_row) / img.height();
    EXPECT_NEAR(hz, 80.0, 6.0);
    EXPECT_TRUE(whalemix::gui::SpectrogramView::render_wav_window(path, 11.0, 1.0, 0).isNull());
    EXPECT_TRUE(whalemix::gui::SpectrogramView::render_wav_window(path, 1.0, 11.0, 3).isNull());
}

TEST(GuiSmokeTest, StreamingAppendMatchesFullRebuild) {
    // Engine runs append rows in many small batches; the incremental path must show exactly what a
    // full rebuild shows, under an active pile + search filter, without re-caching earlier rows.
    std::vector<whalemix::Row> rows;
    for (int i = 0; i < 5000; ++i) {
        whalemix::Row r;
        r.sample_id = "file" + std::to_string(i / 60) + ".wav";
        r.candidate_id = "ev" + std::to_string(i);
        r.picture_start_s = std::to_string(i % 60 * 5) + ".000";
        r.picture_end_s = std::to_string(i % 60 * 5 + 10) + ".156";
        r.bat = (i % 7 == 0) ? "0.950" : "0.010";
        r.mix_label = r.auto_label = (i % 7 == 0) ? "call" : "not-call";
        r.mix_reason = r.auto_reason = "blue_a";
        rows.push_back(r);
    }
    whalemix::gui::ScoreTable full;
    full.resize(900, 600);
    full.set_filter("call", "file1");
    full.set_rows(rows, true);

    whalemix::gui::ScoreTable streamed;
    streamed.resize(900, 600);
    streamed.set_filter("call", "file1");
    std::vector<whalemix::Row> growing;
    size_t shown = 0;
    for (size_t i = 0; i < rows.size(); i += 37) {
        const size_t end = std::min(rows.size(), i + 37);
        growing.insert(growing.end(), rows.begin() + static_cast<std::ptrdiff_t>(i),
                       rows.begin() + static_cast<std::ptrdiff_t>(end));
        streamed.append_rows(growing, shown, true);
        shown = growing.size();
        ASSERT_EQ(streamed.row_count(), shown);
    }
    EXPECT_EQ(streamed.visible_indices(), full.visible_indices());
    EXPECT_GT(streamed.visible_count(), 0);
    // An out-of-step append falls back to a full rebuild instead of duplicating rows.
    streamed.append_rows(rows, 10, true);
    EXPECT_EQ(streamed.row_count(), rows.size());
    EXPECT_EQ(streamed.visible_indices(), full.visible_indices());
}

namespace {
std::vector<whalemix::Row> unified_rows() {
    std::vector<whalemix::Row> rows;
    for (int i = 0; i < 40; ++i) {
        whalemix::Row r;
        r.sample_id = "00001_HYD_20170203.wav";
        r.candidate_id = "ev" + std::to_string(1000 + i);
        r.picture_start_s = std::to_string(i * 5) + ".000";
        r.picture_end_s = std::to_string(i * 5 + 10) + ".156";
        r.audio_path = "/nonexistent/00001_HYD_20170203.wav";
        const bool call = (i % 5 == 0);
        r.bat = call ? "0.969" : "0.005";
        r.mix_label = r.auto_label = call ? "call" : "not-call";
        r.mix_reason = r.auto_reason = call ? "blue_a" : "all assessable whale scores below threshold";
        rows.push_back(r);
    }
    return rows;
}
}  // namespace

TEST(GuiRedesignTest, UnifiedTableDropsBafAndLayaColumnsAndPadsReason) {
    whalemix::gui::ScoreTable t;
    t.resize(1150, 600);
    t.show();
    QApplication::processEvents();
    t.set_rows(unified_rows(), true);
    ASSERT_TRUE(t.unified_mode());
    ASSERT_EQ(t.column_count(), 5);
    QStringList titles;
    for (int i = 0; i < t.column_count(); ++i) titles << t.column_title(i);
    EXPECT_EQ(titles, QStringList({"WINDOW", "TIME", "CONFIDENCE", "REASON", "LABEL"}));
    EXPECT_FALSE(titles.contains("BAF"));
    EXPECT_FALSE(titles.contains("LAYA"));
    EXPECT_EQ(t.reason_display(0).toStdString(), "Blue whale A-call (\u2265 89.7% threshold)");
    EXPECT_EQ(t.reason_display(1).toStdString(), "No call (< 89.7% threshold)");
    // At full width the longest call / not-call REASON text plus 16px padding each side fits.
    QFontMetrics fm(whalemix::fonts::body(13));
    EXPECT_GE(t.column_span(3), fm.horizontalAdvance(t.reason_display(0)) + 32);
    EXPECT_EQ(t.column_left(4), t.column_left(3) + t.column_span(3));
    // The scrollbar lives inside the rounded card: below the header, inset from the right border.
    auto* bar = t.overlay_scroll_bar();
    ASSERT_NE(bar, nullptr);
    EXPECT_TRUE(bar->isVisible());
    EXPECT_EQ(bar->parentWidget(), t.viewport());
    EXPECT_GE(bar->geometry().top(), 34);
    EXPECT_LE(bar->geometry().right(), t.viewport()->width() - 1 - 4);
    EXPECT_LE(bar->geometry().bottom(), t.viewport()->height() - 1 - 4);
    EXPECT_FALSE(t.verticalScrollBar()->isVisible());
    // With the inspector open the table is ~760px wide: REASON still fits and LABEL stays visible.
    t.resize(760, 600);
    QApplication::processEvents();
    // Inspector open: at least the medium variant ("Blue whale A-call (>= 89.7%)") still fits.
    const QStringList v = whalemix::gui::display::reason_variants(unified_rows()[0], 0.897);
    EXPECT_GE(t.column_span(3), fm.horizontalAdvance(v[1]) + 32);
    const QStringList nv = whalemix::gui::display::reason_variants(unified_rows()[1], 0.897);
    EXPECT_GE(t.column_span(3), fm.horizontalAdvance(nv[0]) + 32);
    EXPECT_LE(t.column_left(4) + 112, t.viewport()->width());
    // Legacy CSV rows keep the seven Blue A + Laya columns.
    std::vector<whalemix::Row> legacy = unified_rows();
    for (auto& r : legacy) r.audio_path.clear();
    t.set_rows(legacy, true);
    EXPECT_FALSE(t.unified_mode());
    EXPECT_EQ(t.column_count(), 7);
}

TEST(GuiRedesignTest, PlainEnglishReasonsKeepExportCodes) {
    auto rows = unified_rows();
    EXPECT_EQ(whalemix::gui::display::why_long(rows[0], 0.897).toStdString(),
              "Blue whale A-call detected \xe2\x80\x94 96.9% confidence (above the 89.7% threshold).");
    EXPECT_EQ(whalemix::gui::display::why_long(rows[1], 0.897).toStdString(),
              "No whale call detected \xe2\x80\x94 0.5% confidence (below the 89.7% threshold).");
    EXPECT_EQ(rows[0].mix_reason, "blue_a");  // display text never leaks into exported rows
    EXPECT_EQ(whalemix::gui::display::eta_text(700).toStdString(), "Est. ~11m 40s left");
}

TEST(GuiRedesignTest, InspectorStartsCollapsedOpensOnClickAndCloses) {
    whalemix::MainWindow window;
    window.resize(1440, 900);
    window.show();
    QApplication::processEvents();
    EXPECT_FALSE(window.inspector_visible());
    window.state()->set_rows(unified_rows(), "/tmp/run/out/results.partial.jsonl");
    window.center_panel()->set_rows(window.state()->rows(), true, 1);
    QApplication::processEvents();
    EXPECT_FALSE(window.inspector_visible());
    // Unified model: no Review pile anywhere (pill, sidebar row, SET LABEL option).
    window.center_panel()->table()->set_unified_mode(true);
    window.center_panel()->sync_mode();
    EXPECT_FALSE(window.center_panel()->review_pill()->isVisible());
    const int table_w_collapsed = window.center_panel()->table()->width();

    // Click the first data row (header is 34px, rows are 49px)
    auto* vp = window.center_panel()->table()->viewport();
    QTest::mouseClick(vp, Qt::LeftButton, Qt::NoModifier, QPoint(200, 34 + 49 + 20));
    QApplication::processEvents();
    EXPECT_TRUE(window.inspector_visible());
    EXPECT_EQ(window.state()->selected_index(), 1);
    EXPECT_LT(window.center_panel()->table()->width(), table_w_collapsed);
    // The inspector fits: the panel is inside the scroll viewport, nothing clipped at the right.
    auto* det = window.detail_panel();
    EXPECT_LE(det->width(), static_cast<QScrollArea*>(window.inspector())->viewport()->width());

    // x button collapses it
    QTest::mouseClick(det, Qt::LeftButton, Qt::NoModifier, det->close_button_rect().center());
    QApplication::processEvents();
    EXPECT_FALSE(window.inspector_visible());

    // Click again, then Esc collapses it
    QTest::mouseClick(vp, Qt::LeftButton, Qt::NoModifier, QPoint(200, 34 + 20));
    QApplication::processEvents();
    EXPECT_TRUE(window.inspector_visible());
    QTest::keyClick(&window, Qt::Key_Escape);
    QApplication::processEvents();
    EXPECT_FALSE(window.inspector_visible());
}

TEST(GuiRedesignTest, InspectorLayoutHasNoOverlapOrClipping) {
    whalemix::gui::DetailPanel det;
    det.resize(380, 900);
    auto rows = unified_rows();
    det.set_row(&rows[0], "");
    EXPECT_TRUE(det.unified_mode());
    // Scrubber thumb (radius 5.5 at the track's left end) stays >= 12px clear of the time label.
    QFontMetrics fm(whalemix::fonts::mono(12, 450));
    const int text_right = det.time_label_rect().x() + fm.horizontalAdvance("0:10 / 0:10");
    EXPECT_GE(det.scrub_track_geometry().x() - 6 - text_right, 12);
    // Close button sits inside the panel with the 24px margin.
    EXPECT_LE(det.close_button_rect().right(), 380 - 24);
    // WHY text wraps inside the 332px content column.
    QFontMetrics fw(whalemix::fonts::body(14));
    const QRect b = fw.boundingRect(QRect(24, 0, 332, 400), Qt::TextWordWrap | Qt::AlignLeft, det.why_display_text());
    EXPECT_LE(b.right(), 356);
    EXPECT_FALSE(det.why_display_text().contains("blue_a"));
    EXPECT_TRUE(det.content_height() > 0);
    QPixmap shot = det.grab();
    EXPECT_FALSE(shot.isNull());
}

TEST(GuiRedesignTest, SidebarProgressEtaAndComputeMode) {
    whalemix::MainWindow window;
    auto* sb = window.sidebar();
    EXPECT_FALSE(sb->progress_visible());
    sb->set_progress(1250, 10000, 700.0, false);
    EXPECT_TRUE(sb->progress_visible());
    EXPECT_EQ(sb->progress_text().toStdString(), "1,250 / 10,000 WAV files \xc2\xb7 12.5%");
    EXPECT_EQ(sb->eta_display().toStdString(), "Est. ~11m 40s left");
    sb->set_progress(10000, 10000, -1.0, true);
    EXPECT_EQ(sb->progress_text().toStdString(), "10,000 / 10,000 WAV files \xc2\xb7 Done");

    sb->resize(240, 792);
    QSignalSpy spy(sb, &whalemix::gui::Sidebar::compute_mode_changed);
    const QString before = sb->compute_mode();
    const int target = before == "hybrid" ? 3 : 1;
    QTest::mouseClick(sb, Qt::LeftButton, Qt::NoModifier, sb->compute_segment_rect(target).center());
    EXPECT_EQ(spy.count(), 1);
    EXPECT_EQ(window.engine_bridge()->device(), sb->compute_mode());
    sb->set_compute_backend("cpu", 2, "no GPU execution provider in this build; running on CPU");
    EXPECT_TRUE(sb->compute_subtitle().startsWith("Active: CPU"));
    // Segments stay inside the 192px column and the four of them do not overlap.
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(sb->compute_segment_rect(i).left(), 24);
        EXPECT_LE(sb->compute_segment_rect(i).right(), 216);
        if (i > 0) EXPECT_GT(sb->compute_segment_rect(i).left(), sb->compute_segment_rect(i - 1).right());
    }
    QTest::mouseClick(sb, Qt::LeftButton, Qt::NoModifier, sb->compute_segment_rect(before == "auto" ? 0 : (before == "gpu" ? 2 : (before == "cpu" ? 3 : 1))).center());
}

TEST(GuiRedesignTest, WavFilesAndTenSecondWindowsLabels) {
    whalemix::fonts::init();
    using whalemix::gui::display::source_counts;
    using whalemix::gui::display::wav_files;
    using whalemix::gui::display::wav_progress;
    using whalemix::gui::display::windows_10s;

    EXPECT_EQ(wav_files(1).toStdString(), "1 WAV file");
    EXPECT_EQ(wav_files(10000).toStdString(), "10,000 WAV files");
    EXPECT_EQ(windows_10s(1).toStdString(), "1 10s window");
    EXPECT_EQ(windows_10s(335450).toStdString(), "335,450 10s windows");
    EXPECT_EQ(wav_progress(1, 1).toStdString(), "1 / 1 WAV file");
    EXPECT_EQ(wav_progress(1920, 10000).toStdString(), "1,920 / 10,000 WAV files");
    EXPECT_EQ(source_counts(2135, 65000, true).toStdString(),
              "2,135 WAV files \xc2\xb7 65,000 10s windows");
    EXPECT_EQ(source_counts(10000, 335450, true).toStdString(),
              "10,000 WAV files \xc2\xb7 335,450 10s windows");
    EXPECT_EQ(source_counts(1, 1, false).toStdString(), "1 WAV file \xc2\xb7 1 picture");

    // "10,000 / 10,000 WAV files" must fit the 192px column now that percent lives on the eyebrow row.
    EXPECT_LE(QFontMetrics(whalemix::fonts::body(13)).horizontalAdvance("10,000 / 10,000 WAV files"), 192);
    EXPECT_LE(QFontMetrics(whalemix::fonts::heading(14, 450)).horizontalAdvance("Total windows")
                  + 8
                  + QFontMetrics(whalemix::fonts::mono(14, 450)).horizontalAdvance("335,450"),
              192);
    EXPECT_LE(QFontMetrics(whalemix::fonts::mono(11, 450, 0.1)).horizontalAdvance("PILES (10S WINDOWS)"), 192);

    // The finished 10k SOURCE hint wraps to two lines in 192px and does not elide any characters.
    {
        QFontMetrics fm(whalemix::fonts::body(14));
        const QString hint = source_counts(10000, 335450, true);
        const QRect bound = fm.boundingRect(QRect(0, 0, 192, 80), Qt::TextWordWrap | Qt::AlignLeft | Qt::AlignTop, hint);
        EXPECT_LE(bound.width(), 192);
        EXPECT_GE(bound.height(), fm.lineSpacing());
        EXPECT_LE(bound.height(), fm.lineSpacing() * 2 + 8);
        EXPECT_FALSE(fm.elidedText(hint, Qt::ElideRight, 192 * 2).endsWith(QChar(0x2026)));
    }

    whalemix::MainWindow window;
    auto* sb = window.sidebar();
    sb->set_unified_model(true, 0.897);
    sb->set_source_info("wav10k", 2135, 65000);
    EXPECT_EQ(sb->source_hint_text().toStdString(), "2,135 WAV files \xc2\xb7 65,000 10s windows");
    sb->set_progress(1, 1, 5.0, false);
    EXPECT_EQ(sb->progress_text().toStdString(), "1 / 1 WAV file \xc2\xb7 100.0%");

    // Center-panel meta uses the same units, including singular forms.
    auto rows = unified_rows();
    window.center_panel()->set_rows({rows[0]}, true, 1);
    EXPECT_EQ(window.center_panel()->meta_text().toStdString(), "1 10S WINDOW FROM 1 WAV FILE");
    window.center_panel()->set_rows(rows, true, 2);
    EXPECT_EQ(window.center_panel()->meta_text().toStdString(), "40 10S WINDOWS FROM 2 WAV FILES");
}

int main(int argc, char* argv[]) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
