#pragma once

#include <QWidget>
#include <QString>
#include <QVariantAnimation>
#include <QTimer>
#include "whalemix/row.h"
#include "SpectrogramView.h"
#include "Chip.h"
#include "Kbd.h"

namespace whalemix::gui {

class DetailPanel : public QWidget {
    Q_OBJECT
public:
    explicit DetailPanel(QWidget* parent = nullptr);

    void set_row(const Row* row, const QString& csv_folder);
    const Row* current_row() const { return has_row_ ? &current_row_data_ : nullptr; }

    void toggle_play();
    void set_playing(bool play);
    bool is_playing() const { return playing_; }

    void cycle_gain();
    void set_gain(int g);
    int gain() const { return gain_; }

    void set_playhead(double fraction);
    double playhead() const { return playhead_; }

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;
    int content_height() const;

signals:
    void label_selected(const QString& label);
    void export_requested();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    QRect option_rect(int index) const; // 0 = call, 1 = not-call, 2 = review
    QRect play_button_rect() const;
    QRect scrub_track_rect() const;
    QRect gain_button_rect(int index) const; // 0 = 1x, 1 = 2x, 2 = 4x, 3 = 8x
    int scores_start_y() const;
    bool is_review() const;

    bool has_row_ = false;
    Row current_row_data_;
    QString csv_folder_;

    SpectrogramView* spectro_view_ = nullptr;

    double bat_target_val_ = 0.0;
    double baf_target_val_ = 0.0;
    double bat_anim_val_ = 0.0;
    double baf_anim_val_ = 0.0;
    QVariantAnimation* score_anim_ = nullptr;

    bool playing_ = false;
    double playhead_ = 0.0;
    double clip_duration_ = 6.0;
    int gain_ = 1; // 1, 2, 4, 8
    QTimer* playback_timer_ = nullptr;

    int hovered_option_ = -1;
    int pressed_option_ = -1;

    int hovered_player_item_ = -1; // 0 = play, 1 = scrub, 2 = 1x, 3 = 2x, 4 = 4x, 5 = 8x
    int pressed_player_item_ = -1;
    bool scrubbing_ = false;
};

} // namespace whalemix::gui
