#include "CenterPanel.h"
#include "Theme.h"
#include "Fonts.h"
#include "Eyebrow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>

namespace whalemix::gui {

CenterPanel::CenterPanel(QWidget* parent) : QWidget(parent) {
    auto* main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(24, 24, 24, 24);
    main_layout->setSpacing(0);

    // 1. Head row (40px, bottom-aligned)
    auto* head_widget = new QWidget(this);
    head_widget->setFixedHeight(40);
    auto* head_layout = new QHBoxLayout(head_widget);
    head_layout->setContentsMargins(0, 0, 0, 0);
    head_layout->setSpacing(0);

    auto* title_label = new QLabel("Score windows", head_widget);
    title_label->setFont(fonts::heading(28));
    title_label->setStyleSheet("color: #0b1640;");
    head_layout->addWidget(title_label, 0, Qt::AlignBottom | Qt::AlignLeft);

    head_layout->addStretch(1);

    meta_label_ = new QLabel("NO SOURCE", head_widget);
    auto* meta_label = meta_label_;
    meta_label->setFont(fonts::mono(11, 450, 0.1));
    meta_label->setStyleSheet("color: #56658f; padding-bottom: 8px; text-transform: uppercase;");
    head_layout->addWidget(meta_label, 0, Qt::AlignBottom | Qt::AlignRight);

    main_layout->addWidget(head_widget);

    // 12px gap
    main_layout->addSpacing(12);

    // 2. Filter row (34px)
    auto* filter_widget = new QWidget(this);
    filter_widget->setFixedHeight(36);
    auto* filter_layout = new QHBoxLayout(filter_widget);
    filter_layout->setContentsMargins(0, 0, 0, 0);
    filter_layout->setSpacing(8);

    auto* filter_eyebrow = new Eyebrow("FILTER BY PILE", false, filter_widget);
    filter_layout->addWidget(filter_eyebrow, 0, Qt::AlignVCenter);

    pill_all_ = new PillButton("All", 0, filter_widget);
    pill_call_ = new PillButton("Call", 0, filter_widget);
    pill_not_call_ = new PillButton("Not-call", 0, filter_widget);
    pill_review_ = new PillButton("Review", 0, filter_widget);

    pill_all_->set_active(true);

    filter_layout->addWidget(pill_all_, 0, Qt::AlignVCenter);
    filter_layout->addWidget(pill_call_, 0, Qt::AlignVCenter);
    filter_layout->addWidget(pill_not_call_, 0, Qt::AlignVCenter);
    filter_layout->addWidget(pill_review_, 0, Qt::AlignVCenter);

    filter_layout->addStretch(1);

    search_box_ = new SearchBox(filter_widget);
    filter_layout->addWidget(search_box_, 0, Qt::AlignVCenter);

    main_layout->addWidget(filter_widget);

    // 17px gap
    main_layout->addSpacing(17);

    // 3. Table
    table_ = new ScoreTable(this);
    table_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    main_layout->addWidget(table_, 1);

    // Connect pill clicks
    connect(pill_all_, &PillButton::clicked, this, [this]() { set_active_pile("all"); });
    connect(pill_call_, &PillButton::clicked, this, [this]() { set_active_pile("call"); });
    connect(pill_not_call_, &PillButton::clicked, this, [this]() { set_active_pile("not-call"); });
    connect(pill_review_, &PillButton::clicked, this, [this]() { set_active_pile("review"); });

    // Connect search
    connect(search_box_, &SearchBox::text_changed, this, [this](const QString& q) {
        table_->set_filter(active_pile_, q);
    });

    // Connect selection
    connect(table_, &ScoreTable::selection_changed, this, &CenterPanel::selection_changed);
    connect(table_, &ScoreTable::row_clicked, this, &CenterPanel::row_clicked);
}

void CenterPanel::set_active_pile(const QString& pile) {
    if (active_pile_ != pile) {
        active_pile_ = pile;
        update_pill_active_states();
        table_->set_filter(active_pile_, search_box_->text());
        emit pile_filter_changed(active_pile_);
    }
}

void CenterPanel::update_pill_active_states() {
    pill_all_->set_active(active_pile_ == "all");
    pill_call_->set_active(active_pile_ == "call");
    pill_not_call_->set_active(active_pile_ == "not-call");
    pill_review_->set_active(active_pile_ == "review");
}

void CenterPanel::set_rows(const std::vector<Row>& rows, bool has_file, int num_sounds) {
    num_rows_ = static_cast<int>(rows.size());
    num_sounds_ = num_sounds;
    call_count_ = not_call_count_ = review_count_ = 0;
    for (const auto& r : rows) {
        if (r.mix_label == "call") call_count_++;
        else if (r.mix_label == "not-call") not_call_count_++;
        else if (r.mix_label == "review") review_count_++;
    }

    pill_all_->set_count(num_rows_);
    pill_call_->set_count(call_count_);
    pill_not_call_->set_count(not_call_count_);
    pill_review_->set_count(review_count_);

    table_->set_rows(rows, has_file);
    update_meta_label();
    update();
}

void CenterPanel::sync_mode() {
    const bool unified = table_->unified_mode();
    pill_review_->setVisible(!unified);
    if (unified && active_pile_ == "review") set_active_pile("all");
}

void CenterPanel::update_meta_label() {
    sync_mode();
    if (!meta_label_) return;
    meta_label_->setText(num_rows_ == 0 ? QString("NO SOURCE")
                                        : QString(table_->unified_mode() ? "%L1 WINDOWS FROM %L2 SOUNDS" : "%L1 PICTURES FROM %L2 SOUNDS")
                                              .arg(num_rows_).arg(num_sounds_));
}

void CenterPanel::append_rows(const std::vector<Row>& rows, size_t from, bool has_file, int num_sounds) {
    if (from != table_->row_count() || from > rows.size()) {
        set_rows(rows, has_file, num_sounds);
        return;
    }
    for (size_t i = from; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (r.mix_label == "call") call_count_++;
        else if (r.mix_label == "not-call") not_call_count_++;
        else if (r.mix_label == "review") review_count_++;
    }
    num_rows_ = static_cast<int>(rows.size());
    num_sounds_ = num_sounds;
    pill_all_->set_count(num_rows_);
    pill_call_->set_count(call_count_);
    pill_not_call_->set_count(not_call_count_);
    pill_review_->set_count(review_count_);
    table_->append_rows(rows, from, has_file);
    update_meta_label();
    update();
}

size_t CenterPanel::shown_rows() const { return table_->row_count(); }

void CenterPanel::update_row(size_t idx, const Row& row, const std::string& old_label) {
    auto bump = [this](const std::string& label, int d) {
        if (label == "call") call_count_ += d;
        else if (label == "not-call") not_call_count_ += d;
        else if (label == "review") review_count_ += d;
    };
    bump(old_label, -1);
    bump(row.mix_label, +1);
    pill_call_->set_count(call_count_);
    pill_not_call_->set_count(not_call_count_);
    pill_review_->set_count(review_count_);
    table_->update_row(idx, row);
    update();
}

void CenterPanel::focus_search() {
    search_box_->focus_input();
}

void CenterPanel::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    // Background paper
    p.fillRect(rect(), theme::kPaper);
}

} // namespace whalemix::gui
