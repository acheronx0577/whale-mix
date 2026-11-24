#include <gtest/gtest.h>
#include "whalemix/table.h"
#include "whalemix/exporter.h"
#include "whalemix/report.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string read_binary_file(const fs::path& p) {
    if (!fs::exists(p)) return "";
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void compare_files_byte_for_byte(const fs::path& actual, const fs::path& golden) {
    ASSERT_TRUE(fs::exists(actual)) << "Actual file does not exist: " << actual;
    ASSERT_TRUE(fs::exists(golden)) << "Golden file does not exist: " << golden;

    std::string act = read_binary_file(actual);
    std::string exp = read_binary_file(golden);

    ASSERT_EQ(act.size(), exp.size()) << "Size mismatch for " << actual.filename().string()
                                      << " (actual: " << act.size() << ", golden: " << exp.size() << ")";
    ASSERT_EQ(act, exp) << "Byte mismatch in " << actual.filename().string();
}

void verify_golden_fixture(const std::string& fixture_name, const fs::path& fixtures_dir, const fs::path& golden_dir) {
    auto input_csv = fixtures_dir / (fixture_name + ".csv");
    ASSERT_TRUE(fs::exists(input_csv)) << "Fixture not found: " << input_csv;

    auto fixture_golden = golden_dir / fixture_name;
    ASSERT_TRUE(fs::exists(fixture_golden)) << "Golden folder not found: " << fixture_golden;

    auto out_dir = fs::temp_directory_path() / ("whalemix_golden_" + fixture_name);
    fs::remove_all(out_dir);

    auto result = whalemix::rows_from_blue_a(input_csv);
    whalemix::export_rows(result.rows, out_dir);

    compare_files_byte_for_byte(out_dir / "decisions.csv", fixture_golden / "decisions.csv");
    compare_files_byte_for_byte(out_dir / "decisions.json", fixture_golden / "decisions.json");
    compare_files_byte_for_byte(out_dir / "report.md", fixture_golden / "report.md");
    compare_files_byte_for_byte(out_dir / "call" / "manifest.txt", fixture_golden / "call" / "manifest.txt");
    compare_files_byte_for_byte(out_dir / "not-call" / "manifest.txt", fixture_golden / "not-call" / "manifest.txt");
    compare_files_byte_for_byte(out_dir / "review" / "manifest.txt", fixture_golden / "review" / "manifest.txt");

    fs::remove_all(out_dir);
}

fs::path find_app_tests_dir() {
#ifdef WHALEMIX_SOURCE_DIR
    fs::path src_dir = WHALEMIX_SOURCE_DIR;
    if (fs::exists(src_dir / "app" / "tests" / "fixtures") && fs::exists(src_dir / "app" / "tests" / "golden")) {
        return src_dir / "app" / "tests";
    }
#endif
    // Current working directory or relative to executable
    std::vector<fs::path> candidates = {
        fs::current_path() / "app" / "tests",
        fs::current_path() / "tests",
        fs::current_path().parent_path() / "app" / "tests",
        fs::current_path().parent_path().parent_path() / "app" / "tests",
        fs::current_path().parent_path().parent_path().parent_path() / "app" / "tests",
        "D:/Code/Projects/Marine-Science/MBARI-Portfolio/Blue-Whale-Call-Detection-Under-Ocean-Noise/app/tests"
    };
    for (const auto& c : candidates) {
        if (fs::exists(c / "fixtures") && fs::exists(c / "golden")) {
            return c;
        }
    }
    return "";
}

} // namespace

TEST(GoldenTest, ByteForByteParity) {
    auto tests_dir = find_app_tests_dir();
    ASSERT_FALSE(tests_dir.empty()) << "Could not locate app/tests directory containing fixtures and golden";

    auto fixtures_dir = tests_dir / "fixtures";
    auto golden_dir = tests_dir / "golden";

    std::vector<std::string> fixtures = {
        "blue_a_scores",
        "bom_crlf",
        "edge_cases",
        "header_only"
    };

    for (const auto& f : fixtures) {
        SCOPED_TRACE("Testing fixture: " + f);
        verify_golden_fixture(f, fixtures_dir, golden_dir);
    }
}

TEST(GoldenTest, RaggedRowFailureMessage) {
    auto tests_dir = find_app_tests_dir();
    ASSERT_FALSE(tests_dir.empty());

    auto ragged_csv = tests_dir / "fixtures" / "ragged.csv";
    auto expected_err_path = tests_dir / "golden" / "ragged" / "expected-error.txt";

    ASSERT_TRUE(fs::exists(ragged_csv));
    ASSERT_TRUE(fs::exists(expected_err_path));

    std::string expected_err = read_binary_file(expected_err_path);
    // Strip trailing newlines or whitespace
    while (!expected_err.empty() && (expected_err.back() == '\n' || expected_err.back() == '\r' || expected_err.back() == ' ')) {
        expected_err.pop_back();
    }

    // Expected minus ", byte: 173"
    std::string byte_sub = ", byte: 173";
    auto pos = expected_err.find(byte_sub);
    if (pos != std::string::npos) {
        expected_err.erase(pos, byte_sub.size());
    }
    // Also remove "ERR: " prefix if present in the expected error file
    if (expected_err.rfind("ERR: ", 0) == 0) {
        expected_err = expected_err.substr(5);
    }

    try {
        whalemix::rows_from_blue_a(ragged_csv);
        FAIL() << "ragged.csv was expected to throw";
    } catch (const std::exception& e) {
        std::string actual_err = e.what();
        EXPECT_EQ(actual_err, expected_err);
    }
}

TEST(GoldenTest, SampleAssertions) {
    auto tests_dir = find_app_tests_dir();
    ASSERT_FALSE(tests_dir.empty());

    auto csv_path = tests_dir / "fixtures" / "blue_a_scores.csv";
    auto result = whalemix::rows_from_blue_a(csv_path);

    // 69 rows
    ASSERT_EQ(result.rows.size(), 69);

    // 6 distinct sample_id; per sample A 7, B 9, C 6, D 20, E 22, F 5
    std::map<std::string, size_t> sample_counts;
    size_t call_count = 0;
    size_t not_call_count = 0;
    size_t review_count = 0;
    size_t agree_count = 0;
    size_t blue_a_count = 0;
    size_t disagree_count = 0;

    const whalemix::Row* call_row = nullptr;
    const whalemix::Row* review_row = nullptr;

    for (const auto& r : result.rows) {
        sample_counts[r.sample_id]++;
        if (r.mix_label == "call") {
            call_count++;
            call_row = &r;
        } else if (r.mix_label == "not-call") {
            not_call_count++;
        } else if (r.mix_label == "review") {
            review_count++;
            review_row = &r;
        }

        if (r.mix_reason == "agree") agree_count++;
        else if (r.mix_reason == "blue-a") blue_a_count++;
        else if (r.mix_reason == "disagree") disagree_count++;
    }

    EXPECT_EQ(sample_counts.size(), 6);
    EXPECT_EQ(sample_counts["sample-a"], 7);
    EXPECT_EQ(sample_counts["sample-b"], 9);
    EXPECT_EQ(sample_counts["sample-c"], 6);
    EXPECT_EQ(sample_counts["sample-d"], 20);
    EXPECT_EQ(sample_counts["sample-e"], 22);
    EXPECT_EQ(sample_counts["sample-f"], 5);

    // call 1 / not-call 67 / review 1
    EXPECT_EQ(call_count, 1);
    EXPECT_EQ(not_call_count, 67);
    EXPECT_EQ(review_count, 1);

    // reasons agree 8, blue-a 60, disagree 1
    EXPECT_EQ(agree_count, 8);
    EXPECT_EQ(blue_a_count, 60);
    EXPECT_EQ(disagree_count, 1);

    // call row is sample-a-004 (bat 0.507124, baf 0.492876, 92.416–98.56, image a.jpg, blue-a)
    ASSERT_NE(call_row, nullptr);
    EXPECT_EQ(call_row->candidate_id, "sample-a-004");
    EXPECT_EQ(call_row->bat, "0.507124");
    EXPECT_EQ(call_row->baf, "0.492876");
    EXPECT_EQ(call_row->picture_start_s, "92.416");
    EXPECT_EQ(call_row->picture_end_s, "98.56");
    EXPECT_EQ(call_row->image_file, "a.jpg");
    EXPECT_EQ(call_row->mix_reason, "blue-a");

    // review row is sample-c-002 (Laya call 0.900000, disagree)
    ASSERT_NE(review_row, nullptr);
    EXPECT_EQ(review_row->candidate_id, "sample-c-002");
    EXPECT_EQ(review_row->laya_choice, "call");
    EXPECT_EQ(review_row->laya_probability, "0.900000");
    EXPECT_EQ(review_row->mix_reason, "disagree");

    // report text checks
    auto summary = whalemix::summarize(result.rows);
    std::string report = whalemix::render_report(summary, nullptr);
    EXPECT_NE(report.find("This report covers 69 pictures."), std::string::npos);
    EXPECT_NE(report.find("1 picture went to call. 67 pictures went to not-call. 1 picture went to review."), std::string::npos);
}
