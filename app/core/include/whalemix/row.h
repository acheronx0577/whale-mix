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
};

} // namespace whalemix
