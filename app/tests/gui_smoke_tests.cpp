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

int main(int argc, char* argv[]) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
