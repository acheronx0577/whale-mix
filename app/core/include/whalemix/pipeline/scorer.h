#pragma once

#include "whalemix/table.h"
#include <filesystem>
#include <string>
#include <vector>

namespace whalemix::pipeline {

struct ScoreRequest {
    std::filesystem::path scores_csv;
    std::vector<std::filesystem::path> images;
};

class Scorer {
public:
    virtual ~Scorer() = default;
    virtual std::string name() const = 0;
    virtual LoadResult score(const ScoreRequest& request) = 0;
};

} // namespace whalemix::pipeline
