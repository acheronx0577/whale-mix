#pragma once

#include <QWidget>
#include <QString>
#include <QPixmap>

namespace whalemix::gui {

class Sidebar : public QWidget {
    Q_OBJECT
public:
    explicit Sidebar(QWidget* parent = nullptr);

    void set_counts(int call, int not_call, int review, int total);
    void set_active_pile(const QString& pile);
    QString active_pile() const { return active_pile_; }

    void set_source_info(const QString& file_name, int num_sounds, int num_pictures);
    void set_has_manual(bool has_manual);

    // Unified WhaleMix light model: plain-English "what it detects" instead of the Laya legend.
    void set_unified_model(bool unified, double threshold);
    bool unified_model() const { return unified_; }

    // Scan progress (files_total == 0 hides the section). eta_s < 0 = still estimating.
    void set_progress(int files_done, int files_total, double eta_s, bool finished);
    QString progress_text() const;
    QString eta_display() const;
    QString source_hint_text() const;
    bool progress_visible() const { return files_total_ > 0; }

    // Compute mode: "auto", "hybrid", "gpu" or "cpu". The subtitle reports the active backend.
    void set_compute_mode(const QString& mode);
    QString compute_mode() const { return compute_mode_; }
    void set_compute_backend(const QString& active, int threads, const QString& note);
    QString compute_subtitle() const;
    QRect compute_segment_rect(int index) const;  // 0 Auto, 1 Hybrid, 2 GPU, 3 CPU

    QSize sizeHint() const override;

signals:
    void pile_filter_changed(const QString& pile);
    void compute_mode_changed(const QString& mode);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    QRect pile_row_rect(int index) const; // 0 = Call, 1 = Not-call, 2 = Review, 3 = Total

    int call_count_ = 0;
    int not_call_count_ = 0;
    int review_count_ = 0;
    int total_count_ = 0;

    QString active_pile_ = "all";
    int hovered_row_ = -1;
    int pressed_row_ = -1;

    QString file_name_;
    int num_sounds_ = 0;
    int num_pictures_ = 0;
    bool has_manual_ = false;

    QPixmap dither_pixmap_;

    int progress_top() const;
    int compute_top() const;
    int source_top() const;

    bool unified_ = false;
    double threshold_ = 0.897;
    int files_done_ = 0;
    int files_total_ = 0;
    double eta_s_ = -1.0;
    bool finished_ = false;
    QString compute_mode_ = "auto";
    QString active_backend_;
    int backend_threads_ = 0;
    QString backend_note_;
    int hovered_segment_ = -1;
    int pressed_segment_ = -1;
};

} // namespace whalemix::gui
