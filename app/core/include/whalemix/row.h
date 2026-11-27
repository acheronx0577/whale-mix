#pragma once

#include <string>

namespace whalemix {

struct Row {
    std::string sample_id;
    std::string candidate_id;
    std::string picture_start_s;
    std::string picture_end_s;
    std::string bat;
    std::string baf;
    std::string sound_label;
    std::string image_file;
    std::string laya_choice;
    std::string laya_probability;
    std::string mix_label;
    std::string mix_reason;

    // Computed auto labels so manual overrides can be restored
    std::string auto_label;
    std::string auto_reason;

    // Engine rows only: the analysed recording, so the detail panel can draw the window's spectrogram.
    std::string audio_path;
    int audio_channel = 0;

    // Optional per-window audio stats from the engine (absent on older CSVs / cached jsonl).
    bool has_audio_stats = false;
    double audio_rms = 0.0;
    double band_10_110 = 0.0;
    double band_all = 0.0;
};

} // namespace whalemix
