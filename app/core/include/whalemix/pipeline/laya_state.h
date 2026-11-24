#pragma once

#include "whalemix/row.h"
#include "whalemix/report.h"
#include <string>

namespace whalemix::pipeline {

std::string state_for_row(const Row& row);
std::string state_for_summary(const Summary& summary);

std::string row_question_instructions();
std::string row_question_criteria_call();
std::string row_question_criteria_not_call();

std::string pattern_question_instructions();
std::string pattern_question_criteria_flat();
std::string pattern_question_criteria_one_spike();
std::string pattern_question_criteria_too_few();

std::string rate_ready_question_instructions();
std::string rate_ready_question_criteria_false();
std::string rate_ready_question_criteria_true();

} // namespace whalemix::pipeline
