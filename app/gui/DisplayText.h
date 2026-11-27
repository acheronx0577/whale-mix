#pragma once

// Plain-English display text for the score table, inspector and sidebar. Display only: the row's
// mix_reason / auto_reason keep their engine codes, so exported CSV/JSON stays schema-compatible.

#include <QLocale>
#include <QString>
#include <QStringList>
#include <cmath>
#include <cstdlib>
#include "whalemix/row.h"

namespace whalemix::gui::display {

inline constexpr double kDefaultBlueAThreshold = 0.897;

// Engine rows come from the unified WhaleMix light model (one Blue A score, no baf / Laya columns).
inline bool is_unified_row(const Row& r) { return !r.audio_path.empty(); }

inline bool parse_score(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

inline QString percent(double v, int decimals = 1) {
    return QString::number(v * 100.0, 'f', decimals) + "%";
}

inline QString confidence_text(const Row& r) {
    double v = 0.0;
    return parse_score(r.bat, &v) ? percent(v) : QString::fromUtf8("\xe2\x80\x94");
}

inline bool is_fixture(const Row& r) { return r.mix_reason.rfind("fixture model", 0) == 0; }

// REASON cell variants from longest to shortest; the table shows the longest that fits its column.
inline QStringList reason_variants(const Row& r, double threshold) {
    const QString thr = percent(threshold);
    if (r.mix_reason == "manual") return {"Set by hand (you changed the label)", "Set by hand"};
    if (is_fixture(r)) return {"Fixture model (untrained)", "Fixture model"};
    if (r.bat.empty()) return {"Not assessable for A-calls", "Not assessable"};
    if (r.mix_label == "call")
        return {QString("Blue whale A-call (\u2265 %1 threshold)").arg(thr),
                QString("Blue whale A-call (\u2265 %1)").arg(thr), QString("A-call \u2265 %1").arg(thr)};
    if (r.mix_label == "review") return {"Needs review (uncertain)", "Needs review"};
    return {QString("No call (< %1 threshold)").arg(thr), QString("No call (< %1)").arg(thr),
            QString("< %1").arg(thr)};
}

// One or two plain sentences for the inspector's WHY THIS LABEL block.
inline QString why_long(const Row& r, double threshold) {
    const QString thr = percent(threshold);
    double v = 0.0;
    const bool has = parse_score(r.bat, &v);
    const QString conf = has ? percent(v) : QString();
    QString base;
    if (is_fixture(r)) {
        base = "Scored by an untrained fixture model. This is a plumbing check, not a real detection.";
    } else if (!has) {
        base = "This recording could not be assessed for blue whale A-calls, so no call was detected.";
    } else if (r.auto_label == "call") {
        base = QString("Blue whale A-call detected \xe2\x80\x94 %1 confidence (above the %2 threshold).").arg(conf, thr);
    } else if (r.auto_label == "review") {
        base = QString("Uncertain \xe2\x80\x94 %1 confidence, so this window was sent to review.").arg(conf);
    } else {
        base = QString("No whale call detected \xe2\x80\x94 %1 confidence (below the %2 threshold).").arg(conf, thr);
    }
    if (r.mix_reason == "manual") {
        const QString label = r.mix_label == "call" ? "Call" : (r.mix_label == "review" ? "Review" : "Not-call");
        base += QString(" You changed the label to %1 by hand.").arg(label);
    }
    return base;
}

inline QString decided_by(const Row& r) {
    if (r.mix_reason == "manual") return "Decided by: you (manual label)";
    return "Decided by: WhaleMix Light (automatic)";
}

inline QString thousands(qint64 n) { return QLocale(QLocale::English).toString(n); }

inline QString wav_file_unit(qint64 n) { return n == 1 ? QString("WAV file") : QString("WAV files"); }

inline QString wav_files(qint64 n) { return thousands(n) + " " + wav_file_unit(n); }

inline QString window_10s_unit(qint64 n) { return n == 1 ? QString("10s window") : QString("10s windows"); }

inline QString windows_10s(qint64 n) { return thousands(n) + " " + window_10s_unit(n); }

// Scan progress count, e.g. "1,250 / 10,000 WAV files" or "1 / 1 WAV file".
inline QString wav_progress(qint64 done, qint64 total) {
    return QString("%1 / %2 %3").arg(thousands(done), thousands(total), wav_file_unit(total));
}

// SOURCE / header counts. Unified scans: "2,135 WAV files · 65,000 10s windows".
inline QString source_counts(qint64 wavs, qint64 rows, bool unified) {
    const QString left = wav_files(wavs);
    const QString right = unified ? windows_10s(rows)
                                  : (thousands(rows) + (rows == 1 ? " picture" : " pictures"));
    return left + QString(" \u00b7 ") + right;
}

inline QString eta_text(double seconds) {
    if (!(seconds >= 0.0) || !std::isfinite(seconds)) return "Est. time left: estimating\xe2\x80\xa6";
    const qint64 s = static_cast<qint64>(std::llround(seconds));
    if (s < 60) return QString("Est. ~%1s left").arg(qMax<qint64>(1, s));
    if (s < 3600) return QString("Est. ~%1m %2s left").arg(s / 60).arg(s % 60, 2, 10, QChar('0'));
    return QString("Est. ~%1h %2m left").arg(s / 3600).arg((s % 3600) / 60, 2, 10, QChar('0'));
}

}  // namespace whalemix::gui::display
