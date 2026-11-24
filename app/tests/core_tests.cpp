#include <gtest/gtest.h>
#include "whalemix/rules.h"

using namespace whalemix;

TEST(RulesTest, SoundLabel) {
    EXPECT_EQ(rules::sound_label(0.507, 0.493), "call");
    EXPECT_EQ(rules::sound_label(0.03, 0.97), "not-call");
    EXPECT_EQ(rules::sound_label(0.5, 0.5), "not-call"); // tie -> not-call
    EXPECT_EQ(rules::sound_label(0.492876, 0.507124), "not-call");
    EXPECT_EQ(rules::sound_label(0.507124, 0.492876), "call");
}

TEST(RulesTest, MixLabel) {
    // Agreement
    auto [l1, r1] = rules::mix_label("call", "call", 0.8);
    EXPECT_EQ(l1, "call");
    EXPECT_EQ(r1, "agree");

    auto [l2, r2] = rules::mix_label("not-call", "not-call", 0.9);
    EXPECT_EQ(l2, "not-call");
    EXPECT_EQ(r2, "agree");

    // Disagreement
    auto [l3, r3] = rules::mix_label("not-call", "call", 0.9);
    EXPECT_EQ(l3, "review");
    EXPECT_EQ(r3, "disagree");

    auto [l4, r4] = rules::mix_label("call", "not-call", 0.7);
    EXPECT_EQ(l4, "review");
    EXPECT_EQ(r4, "disagree");

    // Probability thresholds: 0.59 -> blue-a, 0.6 -> confident
    auto [l5, r5] = rules::mix_label("not-call", "call", 0.59);
    EXPECT_EQ(l5, "not-call");
    EXPECT_EQ(r5, "blue-a");

    auto [l6, r6] = rules::mix_label("not-call", "call", 0.6);
    EXPECT_EQ(l6, "review");
    EXPECT_EQ(r6, "disagree");

    auto [l7, r7] = rules::mix_label("call", "call", 0.6);
    EXPECT_EQ(l7, "call");
    EXPECT_EQ(r7, "agree");

    // Unknown choice at 0.9 -> review/disagree
    auto [l8, r8] = rules::mix_label("call", "maybe", 0.9);
    EXPECT_EQ(l8, "review");
    EXPECT_EQ(r8, "disagree");

    // Empty choice with probability -> blue-a
    auto [l9, r9] = rules::mix_label("call", "", 0.95);
    EXPECT_EQ(l9, "call");
    EXPECT_EQ(r9, "blue-a");

    // Empty sound label falls back to not-call
    auto [l10, r10] = rules::mix_label("", "call", 0.8);
    EXPECT_EQ(l10, "review");
    EXPECT_EQ(r10, "disagree");

    auto [l11, r11] = rules::mix_label("", "not-call", 0.8);
    EXPECT_EQ(l11, "not-call");
    EXPECT_EQ(r11, "agree");
}

#include "whalemix/csv.h"
#include "whalemix/table.h"
#include <fstream>

TEST(CsvTest, BasicParsingAndEscapes) {
    std::string csv_data = "col1,col2,\"col,3\",\"col \"\"4\"\"\"\nval1,val2,\"line\nbreak\",val4\r\n";
    auto records = csv::parse_csv(csv_data);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(records[0].fields.size(), 4);
    EXPECT_EQ(records[0].fields[2], "col,3");
    EXPECT_EQ(records[0].fields[3], "col \"4\"");
    EXPECT_EQ(records[1].fields[2], "line\nbreak");
    EXPECT_EQ(records[1].fields[3], "val4");
}

TEST(CsvTest, BomAndCrlf) {
    std::string bom_crlf = "\xEF\xBB\xBFheader1,header2\r\nval1,val2\r\n";
    auto records = csv::parse_csv(bom_crlf);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(records[0].fields[0], "header1");
    EXPECT_EQ(records[1].fields[0], "val1");
}

TEST(TableTest, AliasesAndFallbacks) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_fallbacks";
    std::filesystem::create_directories(dir);
    auto file_path = dir / "test.csv";
    {
        std::ofstream f(file_path, std::ios::binary);
        f << "sample_id,proposal_start_s,proposal_end_s,bat_score,baf_score,image_file\n";
        f << "s-1,10.5,15.5,0.700000,0.300000,\"img,1.jpg\"\n";
    }

    auto res = rows_from_blue_a(file_path);
    ASSERT_EQ(res.rows.size(), 1);
    EXPECT_EQ(res.rows[0].picture_start_s, "10.5");
    EXPECT_EQ(res.rows[0].picture_end_s, "15.5");
    EXPECT_EQ(res.rows[0].bat, "0.700000");
    EXPECT_EQ(res.rows[0].baf, "0.300000");
    EXPECT_EQ(res.rows[0].sound_label, "call");
    EXPECT_EQ(res.rows[0].image_file, "img,1.jpg");

    std::filesystem::remove_all(dir);
}

TEST(TableTest, SkippedRowsAndErrors) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_skips";
    std::filesystem::create_directories(dir);
    auto file_path = dir / "test.csv";
    {
        std::ofstream f(file_path, std::ios::binary);
        f << "sample_id,status,bat,baf\n";
        f << "s-1,scored,0.5,0.4\n";
        f << "s-2,not_scored: edge,0.5,0.4\n";
        f << "s-3,scored, 0.4,0.5\n"; // space before 0.4 -> skipped
        f << "s-4,scored,NaN,0.5\n"; // NaN -> skipped
        f << "s-5,scored,inf,0.5\n"; // inf -> skipped
    }

    auto res = rows_from_blue_a(file_path);
    EXPECT_EQ(res.rows.size(), 1);
    EXPECT_EQ(res.skipped_not_scored, 1);
    EXPECT_EQ(res.skipped_bad_score, 3);

    std::filesystem::remove_all(dir);
}

TEST(TableTest, RaggedRowError) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_ragged";
    std::filesystem::create_directories(dir);
    auto file_path = dir / "test.csv";
    {
        std::ofstream f(file_path, std::ios::binary);
        f << "col1,col2,col3\n";
        f << "val1,val2,val3\n";
        f << "val1,val2\n"; // ragged
    }

    try {
        rows_from_blue_a(file_path);
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("CSV error: record 2 (line: 3): found record with 2 fields, but the previous record has 3 fields"), std::string::npos);
    }

    std::filesystem::remove_all(dir);
}

TEST(TableTest, EmptyAndMissingFile) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_empty";
    std::filesystem::create_directories(dir);
    auto empty_path = dir / "empty.csv";
    {
        std::ofstream f(empty_path, std::ios::binary);
    }

    EXPECT_THROW(rows_from_blue_a(empty_path), std::runtime_error);
    EXPECT_THROW(rows_from_blue_a(dir / "non_existent.csv"), std::runtime_error);

    std::filesystem::remove_all(dir);
}

TEST(TableTest, SavedLayaFormatting) {
    Row r1;
    r1.sound_label = "call";
    r1.laya_choice = "call";
    r1.laya_probability = "abc"; // bad prob -> 0.000000, blue-a
    apply_saved_laya(r1);
    EXPECT_EQ(r1.laya_probability, "0.000000");
    EXPECT_EQ(r1.mix_label, "call");
    EXPECT_EQ(r1.mix_reason, "blue-a");

    Row r2;
    r2.sound_label = "call";
    r2.laya_choice = "call";
    r2.laya_probability = ""; // empty prob -> 0.000000, blue-a
    apply_saved_laya(r2);
    EXPECT_EQ(r2.laya_probability, "0.000000");
    EXPECT_EQ(r2.mix_label, "call");
    EXPECT_EQ(r2.mix_reason, "blue-a");
}

TEST(ManualOverrideTest, SetNoopClear) {
    Row row;
    row.sound_label = "call";
    row.laya_choice = "call";
    row.laya_probability = "0.800000";
    apply_saved_laya(row);
    EXPECT_EQ(row.mix_label, "call");
    EXPECT_EQ(row.mix_reason, "agree");
    EXPECT_EQ(row.auto_label, "call");
    EXPECT_EQ(row.auto_reason, "agree");

    // No-op
    set_label(row, "call");
    EXPECT_EQ(row.mix_label, "call");
    EXPECT_EQ(row.mix_reason, "agree");

    // Override to review
    set_label(row, "review");
    EXPECT_EQ(row.mix_label, "review");
    EXPECT_EQ(row.mix_reason, "manual");
    EXPECT_EQ(row.sound_label, "call");
    EXPECT_EQ(row.laya_choice, "call");
    EXPECT_EQ(row.laya_probability, "0.800000");

    // Clear override
    clear_override(row);
    EXPECT_EQ(row.mix_label, "call");
    EXPECT_EQ(row.mix_reason, "agree");
}

#include "whalemix/exporter.h"
#include "whalemix/json.h"

TEST(ExportTest, BasicExportFormats) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_export";
    std::filesystem::remove_all(dir);

    Row r1;
    r1.sample_id = "sample-a";
    r1.candidate_id = "sample-a-004";
    r1.picture_start_s = "92.416";
    r1.picture_end_s = "98.56";
    r1.bat = "0.507124";
    r1.baf = "0.492876";
    r1.sound_label = "call";
    r1.image_file = "a.jpg";
    r1.laya_choice = "call";
    r1.laya_probability = "0.800000";
    r1.mix_label = "call";
    r1.mix_reason = "agree";

    auto counts = export_rows({r1}, dir);
    EXPECT_EQ(counts.call, 1);
    EXPECT_EQ(counts.not_call, 0);
    EXPECT_EQ(counts.review, 0);
    EXPECT_EQ(counts.rows, 1);

    EXPECT_TRUE(std::filesystem::exists(dir / "decisions.csv"));
    EXPECT_TRUE(std::filesystem::exists(dir / "decisions.json"));
    EXPECT_TRUE(std::filesystem::exists(dir / "call" / "manifest.txt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "not-call" / "manifest.txt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "review" / "manifest.txt"));
    EXPECT_TRUE(std::filesystem::exists(dir / "report.md"));

    // Check call manifest has line
    {
        std::ifstream call_f(dir / "call" / "manifest.txt", std::ios::binary);
        std::string call_m((std::istreambuf_iterator<char>(call_f)), std::istreambuf_iterator<char>());
        EXPECT_EQ(call_m, "sample-a-004\tsample-a\t92.416\t98.56\n");
    }

    // Check not-call manifest is empty (0 bytes)
    EXPECT_EQ(std::filesystem::file_size(dir / "not-call" / "manifest.txt"), 0);

    // Check json format (no trailing newline)
    {
        std::ifstream json_f(dir / "decisions.json", std::ios::binary);
        std::string json_str((std::istreambuf_iterator<char>(json_f)), std::istreambuf_iterator<char>());
        ASSERT_FALSE(json_str.empty());
        EXPECT_EQ(json_str.back(), ']'); // no trailing \n
    }

    std::filesystem::remove_all(dir);
}

TEST(JsonTest, Escaping) {
    EXPECT_EQ(json::escape_string("hello \"world\""), "hello \\\"world\\\"");
    EXPECT_EQ(json::escape_string("line1\nline2\r\t\\"), "line1\\nline2\\r\\t\\\\");
    EXPECT_EQ(json::escape_string("\x01\x1f"), "\\u0001\\u001f");
}

#include "whalemix/report.h"

TEST(ReportTest, RustParityWithReadingAndNoise) {
    Row r1;
    r1.sample_id = "sample-a";
    r1.candidate_id = "sample-a-004";
    r1.picture_start_s = "92.416";
    r1.picture_end_s = "98.56";
    r1.bat = "0.507124";
    r1.baf = "0.492876";
    r1.sound_label = "call";
    r1.laya_choice = "call";
    r1.laya_probability = "0.800000";
    r1.mix_label = "call";
    r1.mix_reason = "agree";

    Row r2;
    r2.sample_id = "sample-c";
    r2.candidate_id = "sample-c-001";
    r2.picture_start_s = "0.256";
    r2.picture_end_s = "3.84";
    r2.bat = "0.032918";
    r2.baf = "0.967082";
    r2.sound_label = "not-call";
    r2.laya_choice = "not-call";
    r2.laya_probability = "0.800000";
    r2.mix_label = "not-call";
    r2.mix_reason = "agree";

    std::unordered_map<std::string, std::string> noise;
    auto summary = summarize({r1, r2}, noise);
    Reading reading{
        "one-spike",
        0.7,
        "no",
        0.2
    };

    std::string text = render_report(summary, &reading);
    auto cover_pos = text.find("This report covers 2 pictures.");
    auto reading_pos = text.find("What Laya said");
    ASSERT_NE(cover_pos, std::string::npos);
    ASSERT_NE(reading_pos, std::string::npos);
    EXPECT_LT(cover_pos, reading_pos);

    EXPECT_NE(text.find("highest call score in Blue A's table is 0.507"), std::string::npos);
    EXPECT_NE(text.find("1 picture went to call."), std::string::npos);
    EXPECT_NE(text.find("None sorted as call."), std::string::npos);
    EXPECT_EQ(text.find("0 pictures"), std::string::npos);
    EXPECT_NE(text.find("Laya said no."), std::string::npos);
    EXPECT_NE(text.find("No 63 Hz level was attached"), std::string::npos);
    EXPECT_NE(text.find("One picture stands out as a call."), std::string::npos);
}

TEST(ReportTest, ManualRowSentence) {
    Row r1;
    r1.sample_id = "sample-a";
    r1.candidate_id = "sample-a-004";
    r1.picture_start_s = "92.416";
    r1.picture_end_s = "98.56";
    r1.bat = "0.507124";
    r1.baf = "0.492876";
    r1.sound_label = "call";
    r1.mix_label = "review";
    r1.mix_reason = "manual";

    auto summary = summarize({r1}, {});
    std::string text = render_report(summary, nullptr);
    EXPECT_NE(text.find("On 1 picture, the label was set by hand."), std::string::npos);
}

#include "whalemix/pipeline/pipeline.h"
#include "whalemix/pipeline/scorer.h"
#include "whalemix/pipeline/sorter.h"
#include "whalemix/pipeline/laya_state.h"

class FakeScorer : public pipeline::Scorer {
public:
    std::string name() const override { return "fake-scorer"; }
    LoadResult score(const pipeline::ScoreRequest&) override {
        LoadResult lr;
        Row r;
        r.sample_id = "sample-fake";
        r.bat = "0.700000";
        r.baf = "0.300000";
        r.sound_label = "call";
        lr.rows.push_back(r);
        return lr;
    }
};

class FakeSorter : public pipeline::Sorter {
public:
    std::string name() const override { return "fake-sorter"; }
    void sort(std::vector<Row>& rows, double) override {
        for (auto& r : rows) {
            r.mix_label = "call";
            r.mix_reason = "fake";
        }
    }
};

TEST(PipelineTest, RunWithFakeClasses) {
    FakeScorer scorer;
    FakeSorter sorter;
    pipeline::Pipeline pipe{scorer, sorter, nullptr};
    pipeline::ScoreRequest req{"dummy.csv", {}};
    auto res = pipeline::run(pipe, req);
    ASSERT_EQ(res.rows.size(), 1);
    EXPECT_EQ(res.rows[0].sample_id, "sample-fake");
    EXPECT_EQ(res.rows[0].mix_label, "call");
    EXPECT_EQ(res.rows[0].mix_reason, "fake");
}

TEST(LayaStateTest, ExactStringsAgainstPythonVerbatim) {
    Row r;
    r.sample_id = "sample-a";
    r.picture_start_s = "92.416";
    r.picture_end_s = "98.56";
    r.bat = "0.507124";
    r.baf = "0.492876";
    r.sound_label = "call";

    std::string state_r = pipeline::state_for_row(r);
    EXPECT_EQ(state_r, "sample sample-a. picture 92.416 to 98.56 seconds. bat 0.507124. baf 0.492876. Blue A label call.");

    EXPECT_EQ(pipeline::row_question_instructions(),
              "A 25-second ocean-sound picture has two scores. bat is the call score. baf is the not-a-call score. Which pile should this picture go in?");
    EXPECT_EQ(pipeline::row_question_criteria_call(), "bat is greater than baf");
    EXPECT_EQ(pipeline::row_question_criteria_not_call(), "baf is greater than or equal to bat");

    EXPECT_EQ(pipeline::pattern_question_instructions(),
              "Using only the counts in the state, how does the call score sit across the pictures?");
    EXPECT_EQ(pipeline::pattern_question_criteria_flat(),
              "the call scores stay in the same low band across samples");
    EXPECT_EQ(pipeline::pattern_question_criteria_one_spike(),
              "one picture is near one half and the rest stay low");
    EXPECT_EQ(pipeline::pattern_question_criteria_too_few(),
              "there are not two call pictures in two different noise levels");

    EXPECT_EQ(pipeline::rate_ready_question_instructions(),
              "Do these counts support a miss rate, a false-alarm rate, or a noise comparison?");
    EXPECT_EQ(pipeline::rate_ready_question_criteria_false(),
              "the table has fewer than two trusted call pictures in each noise level");
    EXPECT_EQ(pipeline::rate_ready_question_criteria_true(),
              "the table has at least two trusted call pictures in a quieter level and a noisier level");
}

#include <cstdlib>
#include "whalemix/pipeline/csv_scorer.h"
#include "whalemix/pipeline/saved_laya_sorter.h"

TEST(CliTest, ExportCliPort) {
    auto dir = std::filesystem::temp_directory_path() / "whalemix_test_export_cli";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    auto scores = dir / "scores.csv";
    {
        std::ofstream f(scores, std::ios::binary);
        f << "sample_id,candidate_id,status,bat_score,baf_score,centered_window_start_s,centered_window_end_s,image_file\n";
        f << "sample-a,sample-a-004,scored,0.507124,0.492876,92.416,98.56,a.jpg\n";
    }

    auto out = dir / "mix";

    // Direct invocation via whalemix_core logic identical to CLI
    whalemix::pipeline::CsvScorer scorer;
    whalemix::pipeline::SavedLayaSorter sorter;
    whalemix::pipeline::Pipeline pipe{scorer, sorter, nullptr};
    whalemix::pipeline::ScoreRequest req{scores, {}};
    auto res = whalemix::pipeline::run(pipe, req);
    whalemix::export_rows(res.rows, out);

    std::ifstream rep_f(out / "report.md", std::ios::binary);
    std::string report((std::istreambuf_iterator<char>(rep_f)), std::istreambuf_iterator<char>());
    EXPECT_NE(report.find("This report covers 1 picture."), std::string::npos);
    EXPECT_NE(report.find("1 picture went to call."), std::string::npos);

    rep_f.close();
    std::filesystem::remove_all(dir);
}
