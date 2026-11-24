file(REMOVE_RECURSE "${TEST_TEMP_DIR}/cli_blue_a")
file(MAKE_DIRECTORY "${TEST_TEMP_DIR}")

# 1. Successful run on blue_a_scores
execute_process(
  COMMAND "${CLI_EXE}" --scores "${SAMPLES_DIR}/blue_a_scores.csv" --out "${TEST_TEMP_DIR}/cli_blue_a"
  RESULT_VARIABLE res
  OUTPUT_VARIABLE cli_out
)
if(NOT res EQUAL 0)
  message(FATAL_ERROR "CLI run on blue_a_scores failed with exit code ${res}")
endif()

# Check output printed out folder
string(STRIP "${cli_out}" cli_out_trimmed)
if(NOT cli_out_trimmed STREQUAL "${TEST_TEMP_DIR}/cli_blue_a")
  message(FATAL_ERROR "CLI did not print out folder path, got: '${cli_out_trimmed}'")
endif()

# Diff decisions.csv against golden
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${TEST_TEMP_DIR}/cli_blue_a/decisions.csv" "${GOLDEN_DIR}/blue_a_scores/decisions.csv"
  RESULT_VARIABLE diff_csv
)
if(NOT diff_csv EQUAL 0)
  message(FATAL_ERROR "decisions.csv differs from golden")
endif()

# Diff decisions.json against golden
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${TEST_TEMP_DIR}/cli_blue_a/decisions.json" "${GOLDEN_DIR}/blue_a_scores/decisions.json"
  RESULT_VARIABLE diff_json
)
if(NOT diff_json EQUAL 0)
  message(FATAL_ERROR "decisions.json differs from golden")
endif()

# Diff report.md against golden
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${TEST_TEMP_DIR}/cli_blue_a/report.md" "${GOLDEN_DIR}/blue_a_scores/report.md"
  RESULT_VARIABLE diff_rep
)
if(NOT diff_rep EQUAL 0)
  message(FATAL_ERROR "report.md differs from golden")
endif()

# 2. Test unknown argument exit 2
execute_process(
  COMMAND "${CLI_EXE}" --unknown-flag
  RESULT_VARIABLE res_unknown
)
if(NOT res_unknown EQUAL 2)
  message(FATAL_ERROR "CLI with unknown flag did not exit with code 2 (got ${res_unknown})")
endif()

# 3. Test missing argument exit 2
execute_process(
  COMMAND "${CLI_EXE}" --scores "${SAMPLES_DIR}/blue_a_scores.csv"
  RESULT_VARIABLE res_missing
)
if(NOT res_missing EQUAL 2)
  message(FATAL_ERROR "CLI with missing --out did not exit with code 2 (got ${res_missing})")
endif()

# 4. Test non-existent file exit 1
execute_process(
  COMMAND "${CLI_EXE}" --scores "non_existent_file.csv" --out "${TEST_TEMP_DIR}/should_not_exist"
  RESULT_VARIABLE res_bad_file
)
if(NOT res_bad_file EQUAL 1)
  message(FATAL_ERROR "CLI with bad file did not exit with code 1 (got ${res_bad_file})")
endif()

message(STATUS "All CLI tests passed successfully!")
