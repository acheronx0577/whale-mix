#include "EngineBridge.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QStandardPaths>
#include <QUuid>

namespace whalemix {

namespace {
constexpr const char* kProtocol = "whalemix-engine/1";
constexpr const char* kSchema = "whalemix-model-bundle/1";
constexpr qint64 kMaxRowsPerRead = 4096;
constexpr qsizetype kMaxPendingRows = 20000;  // bounded hand-off cache; the file stays authoritative
}  // namespace

QList<QJsonObject> EngineLineParser::feed(const QByteArray& bytes) {
    QList<QJsonObject> out;
    qsizetype start = 0;
    while (start < bytes.size()) {
        qsizetype nl = bytes.indexOf('\n', start);
        QByteArray piece = bytes.mid(start, nl < 0 ? -1 : nl - start);
        start = nl < 0 ? bytes.size() : nl + 1;
        if (!skipping_) {
            pending_.append(piece);
            if (pending_.size() > max_line_) {
                pending_.clear();
                skipping_ = true;
                ++dropped_;
            }
        }
        if (nl < 0) break;  // partial line; wait for more bytes
        if (skipping_) {
            skipping_ = false;
            continue;
        }
        QByteArray line = pending_.trimmed();
        pending_.clear();
        if (line.isEmpty()) continue;
        QJsonParseError err{};
        QJsonDocument doc = QJsonDocument::fromJson(line, &err);
        if (err.error != QJsonParseError::NoError || !doc.isObject()) {
            ++invalid_;
            continue;
        }
        out.append(doc.object());
    }
    return out;
}

EngineBridge::EngineBridge(QObject* parent) : EngineBridge(Options{}, parent) {}

EngineBridge::EngineBridge(Options options, QObject* parent) : QObject(parent), options_(std::move(options)) {}

EngineBridge::~EngineBridge() {
    if (process_ && process_->state() != QProcess::NotRunning) {
        send("cancel\n");
        if (!process_->waitForFinished(2000)) {
            process_->kill();
            process_->waitForFinished(1000);
        }
    }
}

QString EngineBridge::discover_bundle_dir() {
    const QString env = qEnvironmentVariable("WHALEMIX_MODEL_BUNDLE");
    if (!env.isEmpty()) return env;
    return QDir(QCoreApplication::applicationDirPath()).filePath("models/acoustic-v1");
}

QString EngineBridge::discover_engine_path() {
    const QString env = qEnvironmentVariable("WHALEMIX_ENGINE");
    if (!env.isEmpty()) return env;
#ifdef Q_OS_WIN
    const QString name = "whale-mix-engine.exe";
#else
    const QString name = "whale-mix-engine";
#endif
    return QDir(QCoreApplication::applicationDirPath()).filePath(name);
}

bool EngineBridge::is_audio_path(const QString& path) {
    static const QStringList kExt = {"wav", "wave", "flac", "aif", "aiff", "mp3", "ogg"};
    return kExt.contains(QFileInfo(path).suffix().toLower());
}

ModelStatus EngineBridge::check_bundle(const QString& bundle_dir, bool allow_fixture) {
    ModelStatus st;
    st.bundle_dir = bundle_dir;
    QFile f(QDir(bundle_dir).filePath("bundle.json"));
    if (bundle_dir.isEmpty() || !f.open(QIODevice::ReadOnly)) {
        st.reason = "no model bundle installed";
        return st;
    }
    if (f.size() > 8 * 1024 * 1024) {
        st.reason = "model bundle manifest is too large";
        return st;
    }
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    const QJsonObject m = doc.object();
    if (m.value("schema").toString() != kSchema) {
        st.reason = "model bundle has an unsupported schema";
        return st;
    }
    st.artifact_kind = m.value("artifact_kind").toString();
    if (m.value("thresholds").toObject().value("blue_a").isDouble()) {
        st.blue_a_threshold = m.value("thresholds").toObject().value("blue_a").toDouble();
    }
    const QString file = m.value("model").toObject().value("file").toString();
    if (file.isEmpty() || file.contains('/') || file.contains('\\') ||
        !QFileInfo(QDir(bundle_dir).filePath(file)).isFile()) {
        st.reason = "model file is missing from the bundle";
        return st;
    }
    if (st.artifact_kind == "fixture-untrained") {
        if (!allow_fixture) {
            st.reason = "installed bundle is an untrained fixture, not a qualified model";
            return st;
        }
    } else if (st.artifact_kind == "selected") {
        if (m.value("qualification").toObject().value("status").toString() != "qualified") {
            st.reason = "installed model has not passed qualification";
            return st;
        }
    } else {
        st.reason = "model bundle has an unknown artifact kind";
        return st;
    }
    // The worker re-validates the SHA-256 of the model file before loading it.
    st.available = true;
    return st;
}

ModelStatus EngineBridge::model_status() const {
    const QString dir = options_.bundle_dir.isEmpty() ? discover_bundle_dir() : options_.bundle_dir;
    return check_bundle(dir, options_.allow_fixture_bundle);
}

bool EngineBridge::running() const { return process_ && process_->state() != QProcess::NotRunning; }

Row row_from_engine_json(const QJsonObject& row) {
    Row r;
    const QString path = row.value("path").toString();
    r.sample_id = (path.isEmpty() ? row.value("source_id").toString() : QFileInfo(path).fileName()).toStdString();
    if (row.value("channel").toInt() > 0) r.sample_id += " ch" + std::to_string(row.value("channel").toInt());
    r.candidate_id = row.value("event_id").toString().toStdString();
    r.picture_start_s = QString::number(row.value("start_s").toDouble(), 'f', 3).toStdString();
    r.picture_end_s = QString::number(row.value("end_s").toDouble(), 'f', 3).toStdString();
    r.audio_path = path.toStdString();
    r.audio_channel = row.value("channel").toInt();
    const QJsonObject scores = row.value("scores").toObject();
    if (scores.value("blue_a").isDouble()) r.bat = QString::number(scores.value("blue_a").toDouble(), 'f', 3).toStdString();
    QString best;
    double best_score = -1.0;
    for (auto it = scores.begin(); it != scores.end(); ++it) {
        if (it.value().isDouble() && it.value().toDouble() > best_score) {
            best_score = it.value().toDouble();
            best = it.key();
        }
    }
    r.sound_label = best.toStdString();
    QString reason = row.value("automatic_reason").toString();
    if (row.value("fixture").toBool()) reason = "fixture model (untrained): " + reason;
    r.auto_label = r.mix_label = row.value("automatic_label").toString().toStdString();
    r.auto_reason = r.mix_reason = reason.toStdString();
    if (row.contains("audio_rms") || row.contains("band_10_110") || row.contains("band_all")) {
        r.has_audio_stats = true;
        r.audio_rms = row.value("audio_rms").toDouble();
        r.band_10_110 = row.value("band_10_110").toDouble();
        r.band_all = row.value("band_all").toDouble();
    }
    return r;
}

void EngineBridge::set_device(const QString& mode) {
    const QString m = mode.toLower();
    if (m == "auto" || m == "hybrid" || m == "gpu" || m == "cpu") options_.device = m;
}

QList<QJsonObject> EngineBridge::take_new_rows() {
    QList<QJsonObject> rows;
    rows.swap(new_rows_);
    return rows;
}

bool EngineBridge::analyze(const QStringList& audio_paths, const QStringList& profiles) {
    if (running()) {
        emit run_finished("rejected", -1, QString());
        return false;
    }
    const ModelStatus st = model_status();
    if (!st.available) {
        emit model_unavailable(st.reason);
        return false;
    }
    const QString engine = options_.engine_path.isEmpty() ? discover_engine_path() : options_.engine_path;
    if (!QFileInfo(engine).isExecutable()) {
        emit model_unavailable("analysis worker is not installed");
        return false;
    }
    QString runs = options_.runs_dir;
    if (runs.isEmpty()) runs = QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation)).filePath("runs");
    run_id_ = QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss") + "-" +
              QUuid::createUuid().toString(QUuid::Id128).left(8);
    run_dir_ = QDir(runs).filePath(run_id_);
    if (!QDir().mkpath(run_dir_)) {
        emit run_finished("failed", -1, QString());
        return false;
    }
    QFile inputs(QDir(run_dir_).filePath("inputs.jsonl"));
    if (!inputs.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        emit run_finished("failed", -1, QString());
        return false;
    }
    int n = 0;
    for (const QString& p : audio_paths) {
        QJsonObject line{{"source_id", QString("src-%1").arg(++n)}, {"path", QFileInfo(p).absoluteFilePath()}};
        inputs.write(QJsonDocument(line).toJson(QJsonDocument::Compact) + "\n");
    }
    inputs.close();

    QStringList args{"--inputs", inputs.fileName(), "--out", QDir(run_dir_).filePath("out"),
                     "--model-bundle", st.bundle_dir, "--threads", QString::number(options_.threads),
                     "--batch", QString::number(options_.batch), "--run-id", run_id_,
                     "--device", options_.device.isEmpty() ? QString("auto") : options_.device};
    for (const QString& p : profiles) args << "--profile" << p;
    if (options_.allow_fixture_bundle) args << "--allow-fixture-bundle";

    parser_ = EngineLineParser();
    partial_offset_ = 0;
    committed_rows_ = 0;
    new_rows_.clear();
    last_seq_ = 0;
    final_status_.clear();
    results_csv_.clear();
    delete process_;
    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::SeparateChannels);
    process_->setStandardErrorFile(QDir(run_dir_).filePath("engine-stderr.log"));
    connect(process_, &QProcess::readyReadStandardOutput, this, &EngineBridge::on_ready_read);
    connect(process_, &QProcess::finished, this, &EngineBridge::on_finished);
    process_->start(engine, args);
    process_started_ever_ = true;
    emit run_started(run_dir_, n);
    return true;
}

void EngineBridge::send(const QByteArray& line) {
    if (running()) process_->write(line);
}
void EngineBridge::pause() { send("pause\n"); }
void EngineBridge::resume() { send("resume\n"); }
void EngineBridge::cancel() { send("cancel\n"); }

void EngineBridge::on_ready_read() {
    rows_pending_ = false;
    for (const QJsonObject& ev : parser_.feed(process_->readAllStandardOutput())) handle(ev);
    // One tail read per stdout chunk, however many results_batch events it carried.
    if (rows_pending_) read_committed_rows();
}

void EngineBridge::handle(const QJsonObject& ev) {
    if (ev.value("protocol").toString() != kProtocol || ev.value("run_id").toString() != run_id_) return;
    const quint64 seq = static_cast<quint64>(ev.value("seq").toDouble());
    if (seq <= last_seq_) return;  // stale or duplicated event
    last_seq_ = seq;
    emit event_received(ev);
    const QString type = ev.value("type").toString();
    if (type == "started") {
        const QString active = ev.value("device_active").toString();
        if (!active.isEmpty()) {
            emit backend_reported(ev.value("device_requested").toString(), active, ev.value("threads").toInt(),
                                  ev.value("device_note").toString());
        }
    } else if (type == "progress") {
        emit progress(ev.value("files_done").toInt(), ev.value("files_total").toInt(),
                      static_cast<qint64>(ev.value("windows_done").toDouble()));
    } else if (type == "results_batch") {
        rows_pending_ = true;
    } else if (type == "file_error") {
        emit file_failed(ev.value("path").toString(), ev.value("kind").toString(), ev.value("message").toString());
    } else if (type == "model_unavailable") {
        emit model_unavailable(ev.value("reason").toString());
    } else if (type == "completed" || type == "canceled") {
        final_status_ = type;
        results_csv_ = ev.value("results_csv").toString();
    } else if (type == "failed") {
        final_status_ = "failed";
    }
}

// Rows are durable in results.partial.jsonl before the worker announces them; read only the new tail.
qint64 EngineBridge::read_committed_rows() {
    QFile f(QDir(run_dir_).filePath("out/results.partial.jsonl"));
    if (!f.open(QIODevice::ReadOnly) || !f.seek(partial_offset_)) return 0;
    qint64 read = 0;
    while (read < kMaxRowsPerRead) {
        const QByteArray line = f.readLine();
        if (line.isEmpty() || !line.endsWith('\n')) break;
        partial_offset_ += line.size();
        const QJsonObject row = QJsonDocument::fromJson(line).object();
        if (row.isEmpty()) continue;
        new_rows_.append(row);
        if (new_rows_.size() > kMaxPendingRows) new_rows_.removeFirst();
        ++read;
    }
    committed_rows_ += read;
    if (read > 0) emit rows_committed(committed_rows_);
    return read;
}

void EngineBridge::on_finished(int exit_code, QProcess::ExitStatus status) {
    on_ready_read();
    while (read_committed_rows() > 0) {
    }
    QString st = final_status_;
    if (st.isEmpty()) st = status == QProcess::CrashExit ? "crashed" : (exit_code == 3 ? "model_unavailable" : "failed");
    emit run_finished(st, exit_code, results_csv_);
}

}  // namespace whalemix
