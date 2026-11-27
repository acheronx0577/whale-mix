#pragma once

// App-side bridge to the `whale-mix-engine` worker (protocol whalemix-engine/1).
//
// The worker is started lazily, only when audio is queued; CSV-only sessions never launch it.
// The process is started with an argument list (no shell). Only a selected bundle whose
// qualification status is "qualified" is used; untrained fixture bundles are refused unless the
// test-only option is set, which the shipping app never does.

#include <QByteArray>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

#include "whalemix/row.h"

namespace whalemix {

// Converts one committed engine result row (results.partial.jsonl) into a score-table row.
// `bat` carries the Blue A score (empty when the recording cannot be assessed for Blue A); the
// highest whale score names the sound. Rows from an untrained fixture bundle say so in mix_reason,
// so fixture output can never look like a model decision.
Row row_from_engine_json(const QJsonObject& row);

// Splits worker stdout into JSON objects. Lines longer than max_line_bytes are discarded (and
// counted) rather than buffered without bound; non-JSON lines are counted and ignored.
class EngineLineParser {
public:
    explicit EngineLineParser(qsizetype max_line_bytes = 512 * 1024) : max_line_(max_line_bytes) {}
    QList<QJsonObject> feed(const QByteArray& bytes);
    int dropped_lines() const { return dropped_; }
    int invalid_lines() const { return invalid_; }

private:
    qsizetype max_line_;
    QByteArray pending_;
    bool skipping_ = false;
    int dropped_ = 0;
    int invalid_ = 0;
};

struct ModelStatus {
    bool available = false;
    QString bundle_dir;
    QString artifact_kind;
    QString reason;  // human-readable when unavailable
    double blue_a_threshold = 0.897;  // bundle thresholds.blue_a (display only)
};

class EngineBridge : public QObject {
    Q_OBJECT
public:
    struct Options {
        QString bundle_dir;   // empty = discover (WHALEMIX_MODEL_BUNDLE, then <app>/models/acoustic-v1)
        QString engine_path;  // empty = discover (WHALEMIX_ENGINE, then <app>/whale-mix-engine)
        QString runs_dir;     // empty = <AppLocalData>/runs
        bool allow_fixture_bundle = false;  // tests only
        int threads = 2;
        int batch = 8;
        QString device = "auto";  // compute mode passed as --device auto|hybrid|gpu|cpu
    };

    explicit EngineBridge(QObject* parent = nullptr);
    EngineBridge(Options options, QObject* parent);
    ~EngineBridge() override;

    static QString discover_bundle_dir();
    static QString discover_engine_path();
    static ModelStatus check_bundle(const QString& bundle_dir, bool allow_fixture);
    static bool is_audio_path(const QString& path);

    ModelStatus model_status() const;
    bool process_started_ever() const { return process_started_ever_; }
    bool running() const;
    QString run_dir() const { return run_dir_; }
    qint64 committed_rows() const { return committed_rows_; }
    QList<QJsonObject> take_new_rows();
    // Compute mode for the next run: "auto", "hybrid", "gpu" or "cpu" (anything else is ignored).
    void set_device(const QString& mode);
    QString device() const { return options_.device; }
    int threads() const { return options_.threads; }

    // Queue audio for analysis. Returns false (with reason via failed/model_unavailable) when the
    // model or worker is unavailable or a run is already active.
    bool analyze(const QStringList& audio_paths, const QStringList& profiles = {});
    void pause();
    void resume();
    void cancel();

signals:
    void model_unavailable(const QString& reason);
    void run_started(const QString& run_dir, int files_total);
    void progress(int files_done, int files_total, qint64 windows_done);
    void rows_committed(qint64 committed_total);
    void file_failed(const QString& path, const QString& kind, const QString& message);
    void run_finished(const QString& status, int exit_code, const QString& results_csv);
    void event_received(const QJsonObject& event);
    // From the worker's "started" event: the backend that actually runs the model.
    void backend_reported(const QString& requested, const QString& active, int threads, const QString& note);

private:
    void on_ready_read();
    void on_finished(int exit_code, QProcess::ExitStatus status);
    void handle(const QJsonObject& event);
    qint64 read_committed_rows();  // returns rows read (at most kMaxRowsPerRead)
    void send(const QByteArray& line);

    Options options_;
    QProcess* process_ = nullptr;
    EngineLineParser parser_;
    bool process_started_ever_ = false;
    QString run_dir_;
    QString run_id_;
    QString final_status_;
    QString results_csv_;
    qint64 partial_offset_ = 0;
    qint64 committed_rows_ = 0;
    QList<QJsonObject> new_rows_;
    quint64 last_seq_ = 0;
    bool rows_pending_ = false;
};

}  // namespace whalemix
