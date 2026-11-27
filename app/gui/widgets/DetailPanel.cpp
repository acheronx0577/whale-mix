#include "DetailPanel.h"
#include "Theme.h"
#include "Fonts.h"
#include "Eyebrow.h"
#include "whalemix/format.h"
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QFontMetrics>
#include <QScrollArea>
#include <QScrollBar>
#include <QApplication>
#include <cmath>
#include <cstdlib>

namespace whalemix::gui {

namespace {

double parse_double(const std::string& str) {
    if (str.empty()) return 0.0;
    char* end = nullptr;
    errno = 0;
    double val = std::strtod(str.c_str(), &end);
    if (end != str.c_str() + str.size() || errno == ERANGE || !std::isfinite(val)) {
        return 0.0;
    }
    return val;
}

} // namespace

DetailPanel::DetailPanel(QWidget* parent) : QWidget(parent) {
    setFixedWidth(380);
    setMouseTracking(true);

    spectro_view_ = new SpectrogramView(this);
    spectro_view_->setGeometry(24, 138, 332, 130);
    spectro_view_->hide();

    score_anim_ = new QVariantAnimation(this);
    score_anim_->setDuration(300);
    score_anim_->setEasingCurve(QEasingCurve::InOutQuad);
    connect(score_anim_, &QVariantAnimation::valueChanged, this, [this](const QVariant& val) {
        double progress = val.toDouble();
        bat_anim_val_ = bat_target_val_ * progress;
        baf_anim_val_ = baf_target_val_ * progress;
        update();
    });

    playback_timer_ = new QTimer(this);
    playback_timer_->setInterval(30);
    connect(playback_timer_, &QTimer::timeout, this, [this]() {
        if (!playing_) return;
        playhead_ += (0.030 / clip_duration_);
        if (playhead_ >= 1.0) {
            playhead_ = 1.0;
            set_playing(false);
        }
        if (spectro_view_) {
            spectro_view_->set_playhead(playhead_);
        }
        update();
    });
}

void DetailPanel::toggle_play() {
    set_playing(!playing_);
}

void DetailPanel::set_playing(bool play) {
    if (playing_ == play) return;
    if (play && playhead_ >= 1.0) {
        playhead_ = 0.0;
    }
    playing_ = play;
    if (playing_) {
        playback_timer_->start();
    } else {
        playback_timer_->stop();
    }
    if (spectro_view_) {
        spectro_view_->set_playhead(playhead_);
    }
    update();
}

void DetailPanel::cycle_gain() {
    int next_g = (gain_ == 1) ? 2 : ((gain_ == 2) ? 4 : ((gain_ == 4) ? 8 : 1));
    set_gain(next_g);
}

void DetailPanel::set_gain(int g) {
    if (gain_ != g) {
        gain_ = g;
        if (spectro_view_) {
            spectro_view_->set_gain(gain_);
        }
        update();
    }
}

void DetailPanel::set_playhead(double fraction) {
    playhead_ = qBound(0.0, fraction, 1.0);
    if (spectro_view_) {
        spectro_view_->set_playhead(playhead_);
    }
    update();
}

bool DetailPanel::is_review() const {
    return has_row_ && current_row_data_.mix_label == "review";
}

int DetailPanel::scores_start_y() const {
    return is_review() ? 398 : 368;
}

QRect DetailPanel::play_button_rect() const {
    return QRect(24, 278, 36, 36);
}

QRect DetailPanel::scrub_track_rect() const {
    return QRect(144, 278 + (36 - 8) / 2, 138, 8);
}

QRect DetailPanel::gain_button_rect(int index) const {
    if (index < 0 || index > 3) return QRect();
    int start_y = is_review() ? 326 : 324;
    int start_x = 84;
    int w = is_review() ? 40 : 42;
    int gap = 4;
    return QRect(start_x + index * (w + gap), start_y, w, 28);
}

void DetailPanel::set_row(const Row* row, const QString& csv_folder) {
    if (!row) {
        has_row_ = false;
        spectro_view_->hide();
        spectro_view_->clear();
        set_playing(false);
        playhead_ = 0.0;
        gain_ = 1;
        setMinimumHeight(120);
        updateGeometry();
        update();
        return;
    }

    has_row_ = true;
    current_row_data_ = *row;
    csv_folder_ = csv_folder;

    set_playing(false);
    playhead_ = 0.0;
    gain_ = 1;

    if (row->image_file.empty() && !row->audio_path.empty()) {
        spectro_view_->set_audio_window(QString::fromStdString(row->audio_path), parse_double(row->picture_start_s),
                                        parse_double(row->picture_end_s), row->audio_channel);
    } else {
        spectro_view_->set_image_info(QString::fromStdString(row->image_file), csv_folder_);
    }
    spectro_view_->set_playhead(0.0);
    spectro_view_->set_gain(1);
    spectro_view_->show();

    bat_target_val_ = parse_double(row->bat);
    baf_target_val_ = parse_double(row->baf);

    score_anim_->stop();
    score_anim_->setStartValue(0.0);
    score_anim_->setEndValue(1.0);
    score_anim_->start();

    setMinimumHeight(content_height());
    updateGeometry();
    update();
}

int DetailPanel::content_height() const {
    if (!has_row_) return 120;
    QRect last_opt = option_rect(2);
    return last_opt.bottom() + 65;
}

QSize DetailPanel::sizeHint() const {
    return QSize(380, content_height());
}

QSize DetailPanel::minimumSizeHint() const {
    return QSize(380, content_height());
}

void DetailPanel::wheelEvent(QWheelEvent* event) {
    if (parentWidget() && parentWidget()->parentWidget()) {
        if (auto* sa = qobject_cast<QScrollArea*>(parentWidget()->parentWidget())) {
            QApplication::sendEvent(sa->verticalScrollBar(), event);
            event->accept();
            return;
        }
    }
    QWidget::wheelEvent(event);
}

QRect DetailPanel::option_rect(int index) const {
    if (!has_row_) return QRect();

    QFont why_font = fonts::body(14);
    QString why_str = QString::fromStdString(format::why_text(current_row_data_));
    QFontMetrics fm_why(why_font);
    QRect bound = fm_why.boundingRect(QRect(24, 0, 332, 300), Qt::TextWordWrap | Qt::AlignLeft, why_str);

    int delta = scores_start_y() - 292;
    int why_y = 433 + delta;
    int reason_y = why_y + bound.height() + 9;
    int set_label_y = reason_y + 18 + 19;
    int options_start_y = set_label_y + 14 + 12;

    if (index >= 0 && index <= 2) {
        return QRect(24, options_start_y + index * (42 + 8), 332, 42);
    }
    return QRect();
}

void DetailPanel::mouseMoveEvent(QMouseEvent* event) {
    if (!has_row_) return;

    if (scrubbing_) {
        QRect r_scrub = scrub_track_rect();
        double frac = static_cast<double>(event->pos().x() - r_scrub.x()) / static_cast<double>(r_scrub.width());
        set_playhead(frac);
        return;
    }

    int hovered = -1;
    for (int i = 0; i < 3; ++i) {
        if (option_rect(i).contains(event->pos())) {
            hovered = i;
            break;
        }
    }

    int hovered_player = -1;
    if (play_button_rect().contains(event->pos())) {
        hovered_player = 0;
    } else if (scrub_track_rect().contains(event->pos())) {
        hovered_player = 1;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (gain_button_rect(i).contains(event->pos())) {
                hovered_player = 2 + i;
                break;
            }
        }
    }

    if (hovered != hovered_option_ || hovered_player != hovered_player_item_) {
        hovered_option_ = hovered;
        hovered_player_item_ = hovered_player;
        bool any_hand = (hovered_option_ >= 0) || (hovered_player_item_ >= 0);
        setCursor(any_hand ? Qt::PointingHandCursor : Qt::ArrowCursor);
        update();
    }
}

void DetailPanel::leaveEvent(QEvent* /*event*/) {
    if (hovered_option_ >= 0 || pressed_option_ >= 0 || hovered_player_item_ >= 0 || pressed_player_item_ >= 0 || scrubbing_) {
        hovered_option_ = -1;
        pressed_option_ = -1;
        hovered_player_item_ = -1;
        pressed_player_item_ = -1;
        scrubbing_ = false;
        setCursor(Qt::ArrowCursor);
        update();
    }
}

void DetailPanel::mousePressEvent(QMouseEvent* event) {
    if (!has_row_ || event->button() != Qt::LeftButton) return;

    if (play_button_rect().contains(event->pos())) {
        pressed_player_item_ = 0;
        toggle_play();
        update();
        return;
    }

    if (scrub_track_rect().contains(event->pos())) {
        scrubbing_ = true;
        pressed_player_item_ = 1;
        QRect r_scrub = scrub_track_rect();
        double frac = static_cast<double>(event->pos().x() - r_scrub.x()) / static_cast<double>(r_scrub.width());
        set_playhead(frac);
        update();
        return;
    }

    for (int i = 0; i < 4; ++i) {
        if (gain_button_rect(i).contains(event->pos())) {
            pressed_player_item_ = 2 + i;
            int gains[4] = {1, 2, 4, 8};
            set_gain(gains[i]);
            update();
            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (option_rect(i).contains(event->pos())) {
            pressed_option_ = i;
            update();
            return;
        }
    }
}

void DetailPanel::mouseReleaseEvent(QMouseEvent* event) {
    if (!has_row_ || event->button() != Qt::LeftButton) return;

    scrubbing_ = false;
    pressed_player_item_ = -1;

    if (pressed_option_ >= 0) {
        int pressed = pressed_option_;
        pressed_option_ = -1;
        update();

        if (option_rect(pressed).contains(event->pos())) {
            const char* label_names[3] = {"call", "not-call", "review"};
            emit label_selected(label_names[pressed]);
        }
    }
}

void DetailPanel::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    int total_h = qMax(height(), content_height());

    // Background white
    p.fillRect(0, 0, width(), total_h, Qt::white);

    // Left border 1px line
    p.setPen(QPen(theme::kLine, 1.0));
    p.drawLine(0, 0, 0, total_h);

    if (!has_row_) {
        // Empty state
        QFont empty_font = fonts::body(14);
        p.setFont(empty_font);
        p.setPen(theme::kSlate);
        p.drawText(QRect(24, 24, 332, 40), Qt::AlignLeft | Qt::AlignVCenter, "Pick a row. Its scores and reason show here.");
        return;
    }

    // 1. Head (22px) at y = 24
    Eyebrow::paint(&p, QRect(24, 24, 200, 22), "SELECTED WINDOW", true);

    // Current label Chip on right
    QString current_label = QString::fromStdString(current_row_data_.mix_label);
    QSize chip_sz = Chip::size_for_label(current_label);
    Chip::paint(&p, QRect(width() - 24 - chip_sz.width(), 24, chip_sz.width(), 22), current_label);

    // 2. Title at y = 54
    QFont title_font = fonts::heading(27);
    p.setFont(title_font);
    p.setPen(theme::kInk);
    QFontMetrics fm_title(title_font);
    QString win_name = QString::fromStdString(current_row_data_.candidate_id.empty() ? current_row_data_.sample_id : current_row_data_.candidate_id);
    p.drawText(QRect(24, 54, 332, 32), Qt::AlignLeft | Qt::AlignVCenter, fm_title.elidedText(win_name, Qt::ElideRight, 332));

    // 3. Sub at y = 97
    QFont sub_font = fonts::mono(13, 450);
    p.setFont(sub_font);
    p.setPen(theme::kSlate);
    QString sub_text = QString("%1 \xc2\xb7 %2").arg(QString::fromStdString(current_row_data_.sample_id)).arg(QString::fromStdString(format::time_text(current_row_data_)));
    p.drawText(QRect(24, 97, 332, 18), Qt::AlignLeft | Qt::AlignVCenter, sub_text);

    // SpectrogramView is placed as a child widget at (24, 138, 332, 130)

    // Player row at y = 278 (height 36px)
    {
        QRect r_play = play_button_rect();
        bool is_play_pressed = (pressed_player_item_ == 0);
        bool is_play_hover = (hovered_player_item_ == 0);

        p.setPen(Qt::NoPen);
        p.setBrush(is_play_pressed ? QColor(15, 36, 130) : (is_play_hover ? theme::kPrimaryHover : theme::kDeep));
        p.drawRoundedRect(r_play, 8.0, 8.0);

        // White play/pause icon
        if (playing_) {
            p.setPen(Qt::NoPen);
            p.setBrush(Qt::white);
            p.fillRect(r_play.x() + 12, r_play.y() + 11, 4, 14, Qt::white);
            p.fillRect(r_play.x() + 20, r_play.y() + 11, 4, 14, Qt::white);
        } else {
            QPolygonF tri;
            tri << QPointF(r_play.x() + 14.5, r_play.y() + 11.5)
                << QPointF(r_play.x() + 24.5, r_play.y() + 18.0)
                << QPointF(r_play.x() + 14.5, r_play.y() + 24.5);
            p.setPen(Qt::NoPen);
            p.setBrush(Qt::white);
            p.drawPolygon(tri);
        }

        // Time text: 0:0X / 0:06
        int cur_sec = static_cast<int>(playhead_ * clip_duration_);
        QString time_str = QString("0:%1 / 0:06").arg(cur_sec, 2, 10, QChar('0'));
        QFont time_font = fonts::mono(12, 450);
        p.setFont(time_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_time(time_font);
        int time_y = r_play.y() + (36 - fm_time.height()) / 2 + fm_time.ascent();
        p.drawText(68, time_y, time_str);

        // Scrub track
        QRect r_scrub = scrub_track_rect();
        p.setPen(QPen(theme::kLine, 1.0));
        p.setBrush(theme::kMist);
        p.drawRoundedRect(QRectF(r_scrub.x(), r_scrub.y(), r_scrub.width(), r_scrub.height()), 4.0, 4.0);

        int fill_scrub_w = static_cast<int>(playhead_ * r_scrub.width());
        if (fill_scrub_w > 0) {
            p.setPen(Qt::NoPen);
            p.setBrush(theme::kDeep);
            p.drawRoundedRect(QRectF(r_scrub.x(), r_scrub.y(), fill_scrub_w, r_scrub.height()), 4.0, 4.0);
        }

        // Thumb circle
        int thumb_x = r_scrub.x() + fill_scrub_w;
        int thumb_y = r_scrub.center().y();
        p.setPen(QPen(Qt::white, 1.5));
        p.setBrush(theme::kDeep);
        p.drawEllipse(QPointF(thumb_x, thumb_y), 5.5, 5.5);

        // Space chip on right
        QSize sp_sz = Kbd::size_for_key("Space", KbdStyle::Default);
        QRect sp_rect(width() - 24 - sp_sz.width(), r_play.y() + (36 - sp_sz.height()) / 2, sp_sz.width(), sp_sz.height());
        Kbd::paint(&p, sp_rect, "Space", KbdStyle::Default);
    }

    // Amplify section at y = 324 (or with mist wrapper if review)
    {
        bool in_review = is_review();
        if (in_review) {
            QRectF r_mist(18.0, 318.0, 344.0, 68.0);
            p.setPen(QPen(theme::kAir, 1.0));
            p.setBrush(theme::kMist);
            p.drawRoundedRect(r_mist, 8.0, 8.0);

            // Note below buttons
            QFont note_font = fonts::body(13);
            p.setFont(note_font);
            p.setPen(theme::kDeep);
            p.drawText(26, 372, "Quiet clip — try amplify");
        }

        // Label AMPLIFY
        int amp_y = in_review ? 326 : 324;
        QFont amp_font = fonts::mono(11, 450, 0.1);
        p.setFont(amp_font);
        p.setPen(in_review ? theme::kDeep : theme::kSlate);
        QFontMetrics fm_amp(amp_font);
        int amp_label_y = amp_y + (28 - fm_amp.height()) / 2 + fm_amp.ascent();
        p.drawText(26, amp_label_y, "AMPLIFY");

        // Segmented buttons: 1x, 2x, 4x, 8x
        int gains[4] = {1, 2, 4, 8};
        for (int i = 0; i < 4; ++i) {
            QRect r_g = gain_button_rect(i);
            bool is_active = (gain_ == gains[i]);
            bool is_hover = (hovered_player_item_ == 2 + i);
            bool is_press = (pressed_player_item_ == 2 + i);

            QRectF rf(r_g.x() + 0.5, r_g.y() + 0.5, r_g.width() - 1.0, r_g.height() - 1.0);
            if (is_active) {
                p.setPen(QPen(theme::kDeep, 1.0));
                p.setBrush(theme::kDeep);
            } else {
                p.setPen(QPen(theme::kLine, 1.0));
                p.setBrush(is_press ? QColor(238, 243, 252) : (is_hover ? theme::kPaper : Qt::white));
            }
            p.drawRoundedRect(rf, 6.0, 6.0);

            QFont g_font = fonts::mono(11, 600, 0.04);
            p.setFont(g_font);
            p.setPen(is_active ? Qt::white : theme::kInk);
            QFontMetrics fm_g(g_font);
            QString g_str = QString("%1x").arg(gains[i]);
            int gx = r_g.x() + (r_g.width() - fm_g.horizontalAdvance(g_str)) / 2;
            int gy = r_g.y() + (r_g.height() - fm_g.height()) / 2 + fm_g.ascent();
            p.drawText(gx, gy, g_str);
        }

        // Kbd A on right
        QSize a_sz = Kbd::size_for_key("A", KbdStyle::Default);
        int a_y = amp_y + (28 - a_sz.height()) / 2;
        QRect a_rect(width() - 24 - a_sz.width(), a_y, a_sz.width(), a_sz.height());
        Kbd::paint(&p, a_rect, "A", KbdStyle::Default);
    }

    int delta = scores_start_y() - 292;

    // 4. BLUE A SCORES
    Eyebrow::paint(&p, QRect(24, 292 + delta, 332, 14), "BLUE A SCORES");

    // Block 1: bat
    int bat_y = 314 + delta;
    {
        // Text row
        QFont bold_font = fonts::mono(13, 450);
        QFont normal_font = fonts::body(14);
        QFont num_font = fonts::mono(13, 450);

        p.setFont(bold_font);
        p.setPen(theme::kInk);
        QFontMetrics fm_bold(bold_font);
        p.drawText(24, bat_y + fm_bold.ascent(), "bat");

        p.setFont(normal_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_norm(normal_font);
        int call_x = 24 + fm_bold.horizontalAdvance("bat ");
        p.drawText(call_x, bat_y + fm_norm.ascent(), "call score");

        // Value on right
        p.setFont(num_font);
        p.setPen(theme::kInk);
        QFontMetrics fm_num(num_font);
        QString val_str = QString::fromStdString(current_row_data_.bat);
        p.drawText(width() - 24 - fm_num.horizontalAdvance(val_str), bat_y + fm_num.ascent(), val_str);

        // 6px Track at y = 341 + delta
        p.setPen(Qt::NoPen);
        p.setBrush(theme::kMist);
        p.drawRoundedRect(QRectF(24.0, 341.0 + delta, 332.0, 6.0), 3.0, 3.0);

        // Fill deep
        int fill_w = static_cast<int>(qMin(1.0, qMax(0.0, bat_anim_val_)) * 332.0);
        if (fill_w > 0) {
            p.setBrush(theme::kDeep);
            p.drawRoundedRect(QRectF(24.0, 341.0 + delta, static_cast<qreal>(fill_w), 6.0), 3.0, 3.0);
        }
    }

    // Block 2: baf
    int baf_y = 359 + delta;
    {
        QFont bold_font = fonts::mono(13, 450);
        QFont normal_font = fonts::body(14);
        QFont num_font = fonts::mono(13, 450);

        p.setFont(bold_font);
        p.setPen(theme::kInk);
        QFontMetrics fm_bold(bold_font);
        p.drawText(24, baf_y + fm_bold.ascent(), "baf");

        p.setFont(normal_font);
        p.setPen(theme::kSlate);
        QFontMetrics fm_norm(normal_font);
        int call_x = 24 + fm_bold.horizontalAdvance("baf ");
        p.drawText(call_x, baf_y + fm_norm.ascent(), "not-a-call score");

        // Value on right
        p.setFont(num_font);
        p.setPen(theme::kInk);
        QFontMetrics fm_num(num_font);
        QString val_str = QString::fromStdString(current_row_data_.baf);
        p.drawText(width() - 24 - fm_num.horizontalAdvance(val_str), baf_y + fm_num.ascent(), val_str);

        // 6px Track at y = 386 + delta
        p.setPen(Qt::NoPen);
        p.setBrush(theme::kMist);
        p.drawRoundedRect(QRectF(24.0, 386.0 + delta, 332.0, 6.0), 3.0, 3.0);

        // Fill air
        int fill_w = static_cast<int>(qMin(1.0, qMax(0.0, baf_anim_val_)) * 332.0);
        if (fill_w > 0) {
            p.setBrush(theme::kAir);
            p.drawRoundedRect(QRectF(24.0, 386.0 + delta, static_cast<qreal>(fill_w), 6.0), 3.0, 3.0);
        }
    }

    // 5. WHY THIS LABEL
    Eyebrow::paint(&p, QRect(24, 411 + delta, 332, 14), "WHY THIS LABEL");

    QFont why_font = fonts::body(14);
    p.setFont(why_font);
    p.setPen(theme::kInk);
    QString why_str = QString::fromStdString(format::why_text(current_row_data_));
    QRect why_bound = p.boundingRect(QRect(24, 433 + delta, 332, 300), Qt::TextWordWrap | Qt::AlignLeft, why_str);
    p.drawText(why_bound, Qt::TextWordWrap | Qt::AlignLeft, why_str);

    // reason: <mix_reason> in mono 13 deep
    int reason_y = why_bound.bottom() + 9;
    QFont reason_font = fonts::mono(13, 450);
    p.setFont(reason_font);
    p.setPen(theme::kDeep);
    p.drawText(QRect(24, reason_y, 332, 18), Qt::AlignLeft | Qt::AlignVCenter, QString("reason: %1").arg(QString::fromStdString(current_row_data_.mix_reason)));

    // 6. SET LABEL
    int set_label_y = reason_y + 18 + 19;
    Eyebrow::paint(&p, QRect(24, set_label_y, 332, 14), "SET LABEL");

    // Option rows
    struct OptionDef {
        QString label;
        QString key;
    };
    OptionDef options[3] = {
        {"call", "C"},
        {"not-call", "N"},
        {"review", "R"}
    };

    for (int i = 0; i < 3; ++i) {
        QRect r = option_rect(i);
        bool is_active = (current_row_data_.mix_label == options[i].label.toStdString());
        bool is_hovered = (hovered_option_ == i);
        bool is_pressed = (pressed_option_ == i);

        if (is_pressed) {
            r = r.adjusted(0, 1, 0, 1);
        }

        // Box
        QRectF box_r(r.x() + 0.5, r.y() + 0.5, r.width() - 1.0, r.height() - 1.0);
        p.setBrush(is_pressed ? QColor(238, 243, 252) : Qt::white);

        if (is_active) {
            p.setPen(QPen(theme::kDeep, 2.0));
        } else if (is_pressed) {
            p.setPen(QPen(theme::kDeep, 1.5));
        } else if (is_hovered) {
            p.setPen(QPen(theme::kAir, 1.0));
        } else {
            p.setPen(QPen(theme::kLine, 1.0));
        }
        p.drawRoundedRect(box_r, 8.0, 8.0);

        // Chip on left at padding 11
        QSize c_sz = Chip::size_for_label(options[i].label);
        Chip::paint(&p, QRect(r.x() + 11, r.y() + (42 - 22) / 2, c_sz.width(), 22), options[i].label);

        // Kbd on right at padding 11
        QSize k_sz = Kbd::size_for_key(options[i].key, KbdStyle::Default);
        Kbd::paint(&p, QRect(r.right() - 11 - k_sz.width(), r.y() + (42 - k_sz.height()) / 2, k_sz.width(), k_sz.height()), options[i].key, KbdStyle::Default);
    }

    // 7. Keys line: 10px below last option row
    QRect last_opt = option_rect(2);
    int keys_y = last_opt.bottom() + 10;
    int cur_k_x = 24;

    QFont keys_font = fonts::body(13);
    p.setFont(keys_font);
    p.setPen(theme::kSlate);
    QFontMetrics fm_keys(keys_font);
    int text_ky = keys_y + (21 - fm_keys.height()) / 2 + fm_keys.ascent();

    QSize sp_k = Kbd::size_for_key("Space", KbdStyle::Default);
    Kbd::paint(&p, QRect(cur_k_x, keys_y, sp_k.width(), sp_k.height()), "Space", KbdStyle::Default);
    cur_k_x += sp_k.width() + 5;
    p.drawText(cur_k_x, text_ky, "play");
    cur_k_x += fm_keys.horizontalAdvance("play") + 8;

    QSize a_k = Kbd::size_for_key("A", KbdStyle::Default);
    Kbd::paint(&p, QRect(cur_k_x, keys_y, a_k.width(), a_k.height()), "A", KbdStyle::Default);
    cur_k_x += a_k.width() + 5;
    p.drawText(cur_k_x, text_ky, "amplify");
    cur_k_x += fm_keys.horizontalAdvance("amplify") + 8;

    QSize j_sz = Kbd::size_for_key("J", KbdStyle::Default);
    Kbd::paint(&p, QRect(cur_k_x, keys_y, j_sz.width(), j_sz.height()), "J", KbdStyle::Default);
    cur_k_x += j_sz.width() + 3;

    QSize k_sz = Kbd::size_for_key("K", KbdStyle::Default);
    Kbd::paint(&p, QRect(cur_k_x, keys_y, k_sz.width(), k_sz.height()), "K", KbdStyle::Default);
    cur_k_x += k_sz.width() + 5;

    p.drawText(cur_k_x, text_ky, "navigate");
    cur_k_x += fm_keys.horizontalAdvance("navigate") + 8;

    QSize e_sz = Kbd::size_for_key("E", KbdStyle::Default);
    Kbd::paint(&p, QRect(cur_k_x, keys_y, e_sz.width(), e_sz.height()), "E", KbdStyle::Default);
    cur_k_x += e_sz.width() + 5;

    p.drawText(cur_k_x, text_ky, "export");
}

} // namespace whalemix::gui
