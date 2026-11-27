// Engine bridge tests: missing-model behaviour, CSV-only sessions never start the worker, protocol
// line parsing, fixture refusal, and (when WHALEMIX_ENGINE + WHALEMIX_FIXTURE_BUNDLE are set) Qt
// responsiveness and cancellation during untrained-fixture inference. Fixture timings are
// plumbing measurements only.
#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDataStream>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>
#include <cmath>
#include <iostream>

#include "EngineBridge.h"
#include "MainWindow.h"
#include "whalemix/exporter.h"

// A fixture-dependent test cannot run: with WHALEMIX_REQUIRE_FIXTURE=1 (CI) that fails instead of skipping.
#define WHALEMIX_REQUIRE_OR_SKIP(ok, msg)                                          \
    do {                                                                            \
        if (!(ok)) {                                                                \
            if (qEnvironmentVariable("WHALEMIX_REQUIRE_FIXTURE") == "1") FAIL() << "required: " << msg; \
            GTEST_SKIP() << msg;                                                    \
        }                                                                           \
    } while (0)

using whalemix::EngineBridge;
using whalemix::EngineLineParser;

namespace {

void write_wav(const QString& path, int rate, double seconds) {
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    const quint32 n = static_cast<quint32>(rate * seconds);
    QDataStream s(&f);
    s.setByteOrder(QDataStream::LittleEndian);
    s.writeRawData("RIFF", 4);
    s << quint32(36 + n * 2);
    s.writeRawData("WAVEfmt ", 8);
    s << quint32(16) << quint16(1) << quint16(1) << quint32(rate) << quint32(rate * 2) << quint16(2) << quint16(16);
    s.writeRawData("data", 4);
    s << quint32(n * 2);
    for (quint32 i = 0; i < n; ++i) {
        const double t = double(i) / rate;
        s << qint16(std::lround(8000.0 * std::sin(2 * 3.141592653589793 * (60.0 * t + 2.0 * t * t))));
    }
}

// Peak resident memory of this (app) process in MiB, or -1 where /proc is unavailable.
double app_peak_rss_mib() {
    QFile f("/proc/self/status");
    if (!f.open(QIODevice::ReadOnly)) return -1;
    for (const QByteArray& line : f.readAll().split('\n')) {
        if (line.startsWith("VmHWM:")) return line.simplified().split(' ').value(1).toDouble() / 1024.0;
    }
    return -1;
}

QString fixture_bundle() { return qEnvironmentVariable("WHALEMIX_FIXTURE_BUNDLE"); }
QString engine_exe() { return qEnvironmentVariable("WHALEMIX_ENGINE"); }

EngineBridge* fixture_bridge(const QString& runs) {
    EngineBridge::Options o;
    o.bundle_dir = fixture_bundle();
    o.engine_path = engine_exe();
    o.runs_dir = runs;
    o.allow_fixture_bundle = true;  // test-only; the shipping app never sets this
    return new EngineBridge(o, nullptr);
}

}  // namespace

TEST(EngineLineParserTest, SplitsChunkedLinesAndBoundsLength) {
    EngineLineParser p(64);
    auto a = p.feed("{\"type\":\"pro");
    EXPECT_TRUE(a.isEmpty());
    auto b = p.feed("gress\",\"seq\":1}\n{\"type\":\"x\"}\nnot json\n");
    ASSERT_EQ(b.size(), 2);
    EXPECT_EQ(b[0].value("type").toString(), "progress");
    EXPECT_EQ(p.invalid_lines(), 1);
    auto c = p.feed(QByteArray(200, 'a') + "\n{\"ok\":1}\n");
    ASSERT_EQ(c.size(), 1);
    EXPECT_EQ(p.dropped_lines(), 1);
}

TEST(EngineBridgeTest, MissingModelKeepsCsvViewingAndNeverStartsWorker) {
    QTemporaryDir tmp;
    whalemix::MainWindow window;
    EngineBridge::Options o;
    o.bundle_dir = tmp.filePath("no-bundle");
    o.engine_path = tmp.filePath("no-engine");
    window.set_engine_bridge(new EngineBridge(o, nullptr));
    const QString wav = tmp.filePath("a.wav");
    write_wav(wav, 8000, 1.0);
    window.ingest_paths({wav});
    EXPECT_TRUE(window.status_bar()->status().startsWith("Model unavailable: no model bundle installed"));
    EXPECT_FALSE(window.engine_bridge()->process_started_ever());
}

TEST(EngineBridgeTest, CsvOnlySessionDoesNotLaunchWorker) {
    whalemix::MainWindow window;
    const QString csv = QString(WHALEMIX_SOURCE_DIR) + "/samples/blue_a_scores.csv";
    window.ingest_paths({csv});
    EXPECT_EQ(window.center_panel()->table()->visible_count(), 69);
    EXPECT_FALSE(window.engine_bridge()->process_started_ever());
    EXPECT_FALSE(window.engine_bridge()->running());
}

TEST(EngineBridgeTest, DefaultDiscoveryRefusesFixtureBundles) {
    WHALEMIX_REQUIRE_OR_SKIP(!fixture_bundle().isEmpty(), "WHALEMIX_FIXTURE_BUNDLE not set");
    auto refused = EngineBridge::check_bundle(fixture_bundle(), false);
    EXPECT_FALSE(refused.available);
    EXPECT_EQ(refused.artifact_kind, "fixture-untrained");
    EXPECT_TRUE(EngineBridge::check_bundle(fixture_bundle(), true).available);
}

TEST(EngineBridgeTest, FixtureInferenceStreamsRowsAndKeepsUiResponsive) {
    WHALEMIX_REQUIRE_OR_SKIP(!fixture_bundle().isEmpty() && !engine_exe().isEmpty(), "fixture engine not configured");
    QTemporaryDir tmp;
    QStringList wavs;
    for (int i = 0; i < 6; ++i) {
        wavs << tmp.filePath(QString("long-%1.wav").arg(i));
        write_wav(wavs.last(), 8000, 300.0);
    }
    whalemix::MainWindow window;
    window.set_engine_bridge(fixture_bridge(tmp.filePath("runs")));
    auto* bridge = window.engine_bridge();

    QElapsedTimer clock;
    qint64 last_tick = -1, max_gap = 0;
    QTimer ticker;
    ticker.setInterval(10);
    QObject::connect(&ticker, &QTimer::timeout, [&] {
        const qint64 now = clock.elapsed();
        if (last_tick >= 0) max_gap = std::max(max_gap, now - last_tick);
        last_tick = now;
    });
    qint64 first_rows_ms = -1, first_progress_ms = -1;
    bool rows_before_finish = false;
    QObject::connect(bridge, &EngineBridge::progress, [&](int, int, qint64) {
        if (first_progress_ms < 0) first_progress_ms = clock.elapsed();
    });
    QObject::connect(bridge, &EngineBridge::rows_committed, [&](qint64) {
        if (first_rows_ms < 0) first_rows_ms = clock.elapsed();
        rows_before_finish = true;
        window.state();  // GUI object still usable mid-run
    });
    QSignalSpy finished(bridge, &EngineBridge::run_finished);
    clock.start();
    ticker.start();
    ASSERT_TRUE(bridge->analyze(wavs));
    ASSERT_TRUE(finished.wait(120000));
    const qint64 total_ms = clock.elapsed();
    ticker.stop();
    EXPECT_EQ(finished.first().at(0).toString(), "completed");
    EXPECT_TRUE(rows_before_finish);
    EXPECT_EQ(bridge->committed_rows(), 6 * 59);
    // Engine results populate the score table, labelled as fixture output.
    ASSERT_EQ(window.state()->rows().size(), static_cast<size_t>(6 * 59));
    EXPECT_NE(window.state()->rows().front().mix_reason.find("fixture model"), std::string::npos);
    EXPECT_LT(max_gap, 250) << "GUI event loop stalled";
    std::cout << "[fixture] gui: first_progress_ms=" << first_progress_ms << " first_rows_ms=" << first_rows_ms
              << " total_ms=" << total_ms << " audio_s=1800 max_event_loop_gap_ms=" << max_gap << " app_peak_rss_mib=" << app_peak_rss_mib()
              << std::endl;
}

TEST(EngineBridgeTest, EngineRowsConvertToScoreTableRows) {
    const QJsonObject row = QJsonDocument::fromJson(R"({
        "event_id": "ab12", "source_id": "s1", "path": "/data/deep/site-a/clip.wav", "channel": 1,
        "start_s": 12.5, "end_s": 37.5,
        "scores": {"blue_a": 0.8126, "blue_b": null, "fin_20hz_pulse": 0.25},
        "automatic_label": "call", "automatic_reason": "blue_a", "fixture": false})").object();
    const whalemix::Row r = whalemix::row_from_engine_json(row);
    EXPECT_EQ(r.sample_id, "clip.wav ch1");
    EXPECT_EQ(r.candidate_id, "ab12");
    EXPECT_EQ(r.picture_start_s, "12.500");
    EXPECT_EQ(r.picture_end_s, "37.500");
    EXPECT_EQ(r.bat, "0.813");
    EXPECT_EQ(r.baf, "");
    EXPECT_EQ(r.sound_label, "blue_a");
    EXPECT_EQ(r.mix_label, "call");
    EXPECT_EQ(r.auto_label, "call");
    EXPECT_EQ(r.mix_reason, "blue_a");

    QJsonObject fixture = row;
    fixture["fixture"] = true;
    fixture["automatic_label"] = "review";
    fixture["automatic_reason"] = "thresholds not-applicable; scores only";
    QJsonObject unscored = fixture;
    unscored["scores"] = QJsonObject{{"blue_a", QJsonValue()}};
    const whalemix::Row f = whalemix::row_from_engine_json(fixture);
    EXPECT_EQ(f.mix_label, "review");
    EXPECT_EQ(f.mix_reason, "fixture model (untrained): thresholds not-applicable; scores only");
    EXPECT_EQ(whalemix::row_from_engine_json(unscored).bat, "");
}

TEST(EngineBridgeTest, CancelIsAcknowledgedAndPartialResultsKept) {
    WHALEMIX_REQUIRE_OR_SKIP(!fixture_bundle().isEmpty() && !engine_exe().isEmpty(), "fixture engine not configured");
    QTemporaryDir tmp;
    QStringList wavs;
    for (int i = 0; i < 4; ++i) {
        wavs << tmp.filePath(QString("c-%1.wav").arg(i));
        write_wav(wavs.last(), 8000, 300.0);
    }
    auto* bridge = fixture_bridge(tmp.filePath("runs"));
    QElapsedTimer since_cancel;
    QObject::connect(bridge, &EngineBridge::rows_committed, [&](qint64) {
        if (!since_cancel.isValid()) {
            since_cancel.start();
            bridge->cancel();
        }
    });
    QSignalSpy finished(bridge, &EngineBridge::run_finished);
    ASSERT_TRUE(bridge->analyze(wavs));
    ASSERT_TRUE(finished.wait(60000));
    const qint64 ack_ms = since_cancel.elapsed();
    EXPECT_EQ(finished.first().at(0).toString(), "canceled");
    EXPECT_EQ(finished.first().at(1).toInt(), 4);
    EXPECT_GT(bridge->committed_rows(), 0);
    EXPECT_LT(bridge->committed_rows(), 4 * 59);
    EXPECT_TRUE(QFile::exists(bridge->run_dir() + "/out/results.partial.csv"));
    EXPECT_LT(ack_ms, 2000);
    std::cout << "[fixture] cancel: process exit " << ack_ms << " ms after cancel request, committed_rows="
              << bridge->committed_rows() << std::endl;
    delete bridge;
}

// Opt-in stress run with a real qualified bundle (never a fixture): drops a folder of audio through
// the same ingestion path as drag-and-drop, then checks the GUI event loop never stalls while rows
// stream into the table and that every committed row is shown. Enable with
// WHALEMIX_STRESS_AUDIO_DIR=<folder> WHALEMIX_MODEL_BUNDLE=<bundle> WHALEMIX_ENGINE=<worker>.
TEST(EngineBridgeTest, QualifiedBundleFolderDropStreamsWithoutStalling) {
    const QString dir = qEnvironmentVariable("WHALEMIX_STRESS_AUDIO_DIR");
    const QString bundle = qEnvironmentVariable("WHALEMIX_MODEL_BUNDLE");
    if (dir.isEmpty() || bundle.isEmpty() || engine_exe().isEmpty()) GTEST_SKIP() << "stress run not configured";
    ASSERT_TRUE(EngineBridge::check_bundle(bundle, false).available) << "bundle is not a qualified release";
    QTemporaryDir tmp;
    whalemix::MainWindow window;
    window.resize(1440, 900);
    window.show();
    EngineBridge::Options o;
    o.bundle_dir = bundle;
    o.engine_path = engine_exe();
    o.runs_dir = tmp.filePath("runs");
    window.set_engine_bridge(new EngineBridge(o, nullptr));
    auto* bridge = window.engine_bridge();

    QElapsedTimer clock;
    qint64 last_tick = -1, max_gap = 0, ticks_over_100 = 0;
    QTimer ticker;
    ticker.setInterval(10);
    QObject::connect(&ticker, &QTimer::timeout, [&] {
        const qint64 now = clock.elapsed();
        if (last_tick >= 0) {
            max_gap = std::max(max_gap, now - last_tick);
            if (now - last_tick > 100) ++ticks_over_100;
        }
        last_tick = now;
    });
    int files_total = 0, last_files_done = 0;
    qint64 first_rows_ms = -1;
    QObject::connect(bridge, &EngineBridge::run_started, [&](const QString&, int n) { files_total = n; });
    QObject::connect(bridge, &EngineBridge::progress, [&](int done, int, qint64) { last_files_done = done; });
    QObject::connect(bridge, &EngineBridge::rows_committed, [&](qint64) {
        if (first_rows_ms < 0) first_rows_ms = clock.elapsed();
    });
    QSignalSpy finished(bridge, &EngineBridge::run_finished);
    clock.start();
    ticker.start();
    window.ingest_paths({dir});
    const int timeout_ms = qEnvironmentVariableIntValue("WHALEMIX_STRESS_TIMEOUT_MS") > 0
                               ? qEnvironmentVariableIntValue("WHALEMIX_STRESS_TIMEOUT_MS") : 3600000;
    ASSERT_TRUE(finished.wait(timeout_ms));
    QCoreApplication::processEvents();
    ticker.stop();
    const qint64 total_ms = clock.elapsed();
    EXPECT_EQ(finished.first().at(0).toString(), "completed");
    const size_t rows = window.state()->rows().size();
    EXPECT_EQ(static_cast<qint64>(rows), bridge->committed_rows());
    EXPECT_EQ(window.center_panel()->shown_rows(), rows);
    const int max_gap_budget = qEnvironmentVariableIntValue("WHALEMIX_STRESS_MAX_GAP_MS") > 0
                                   ? qEnvironmentVariableIntValue("WHALEMIX_STRESS_MAX_GAP_MS") : 250;
    EXPECT_LT(max_gap, max_gap_budget) << "GUI event loop stalled";
    std::cout << "[stress] files=" << files_total << " files_done=" << last_files_done << " rows=" << rows
              << " first_rows_ms=" << first_rows_ms << " total_ms=" << total_ms
              << " max_event_loop_gap_ms=" << max_gap << " gaps_over_100ms=" << ticks_over_100
              << " app_peak_rss_mib=" << app_peak_rss_mib() << std::endl;
}

// Opt-in: load a real engine results.partial.jsonl (e.g. a 10,000-file run) into the window and
// time the interactive features on that table size. WHALEMIX_STRESS_RESULTS=<results.partial.jsonl>.
TEST(EngineBridgeTest, LargeEngineTableFeaturesStayInteractive) {
    const QString path = qEnvironmentVariable("WHALEMIX_STRESS_RESULTS");
    if (path.isEmpty()) GTEST_SKIP() << "WHALEMIX_STRESS_RESULTS not set";
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly));
    std::vector<whalemix::Row> rows;
    while (!f.atEnd()) {
        const QByteArray line = f.readLine();
        const QJsonObject obj = QJsonDocument::fromJson(line).object();
        if (!obj.isEmpty()) rows.push_back(whalemix::row_from_engine_json(obj));
    }
    ASSERT_FALSE(rows.empty());
    const size_t n = rows.size();
    whalemix::MainWindow window;
    window.resize(1440, 900);
    window.show();
    QElapsedTimer t;
    t.start();
    window.state()->set_rows(std::move(rows), path);
    window.center_panel()->set_rows(window.state()->rows(), true, 0);
    const qint64 load_ms = t.restart();
    window.center_panel()->set_active_pile("call");
    const int calls = window.center_panel()->table()->visible_count();
    const qint64 filter_ms = t.restart();
    window.center_panel()->set_active_pile("all");
    window.center_panel()->table()->select_first_call_or_first();
    QCoreApplication::processEvents();
    const qint64 select_ms = t.restart();
    window.set_manual_label("review");
    const qint64 label_ms = t.restart();
    window.undo_override();
    const qint64 undo_ms = t.restart();
    QTemporaryDir out;
    whalemix::export_rows(window.state()->rows(), out.path().toStdString());
    const qint64 export_ms = t.restart();
    QFile csv(out.filePath("decisions.csv"));
    ASSERT_TRUE(csv.open(QIODevice::ReadOnly));
    qint64 lines = 0;
    while (!csv.atEnd()) { csv.readLine(); ++lines; }
    EXPECT_EQ(lines, static_cast<qint64>(n) + 1);
    EXPECT_GT(calls, 0);
    std::cout << "[features] rows=" << n << " calls=" << calls << " load_ms=" << load_ms
              << " filter_call_ms=" << filter_ms << " select_ms=" << select_ms << " manual_label_ms=" << label_ms
              << " undo_ms=" << undo_ms << " export_ms=" << export_ms << " decisions_csv_lines=" << lines << std::endl;
}
