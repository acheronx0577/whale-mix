#pragma once

// Plain-English display text for the score table, inspector and sidebar. Display only: the row's
// mix_reason / auto_reason keep their engine codes, so exported CSV/JSON stays schema-compatible.

#include <QLocale>
#include <QString>
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

// Short label for the table's REASON column.
inline QString reason_short(const Row& r) {
    if (r.mix_reason == "manual") return "Set by hand";
    if (is_fixture(r)) return "Fixture model (untrained)";
    if (r.mix_label == "call") return "Blue whale A-call";
    if (r.mix_label == "review") return "Needs review";
    if (r.bat.empty()) return "Not assessable";
    return "Below threshold";
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

inline QString eta_text(double seconds) {
    if (!(seconds >= 0.0) || !std::isfinite(seconds)) return "Est. time left: estimating\xe2\x80\xa6";
    const qint64 s = static_cast<qint64>(std::llround(seconds));
    if (s < 60) return QString("Est. ~%1s left").arg(qMax<qint64>(1, s));
    if (s < 3600) return QString("Est. ~%1m %2s left").arg(s / 60).arg(s % 60, 2, 10, QChar('0'));
    return QString("Est. ~%1h %2m left").arg(s / 3600).arg((s % 3600) / 60, 2, 10, QChar('0'));
}

}  // namespace whalemix::gui::display
