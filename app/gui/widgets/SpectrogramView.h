#pragma once

#include <QWidget>
#include <QString>
#include <QImage>
#include <QFutureWatcher>
#include <map>
#include <list>
#include <mutex>

namespace whalemix::gui {

class SpectrogramView : public QWidget {
    Q_OBJECT
public:
    explicit SpectrogramView(QWidget* parent = nullptr);

    void set_image_info(const QString& image_file, const QString& csv_folder);
    void set_playhead(double fraction);
    void set_gain(int gain);
    double playhead() const { return playhead_; }
    int gain() const { return gain_; }
    void clear();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void load_image_async(const QString& resolved_path);
    QString resolve_image_path(const QString& image_file, const QString& csv_folder) const;

    QString image_file_;
    QString csv_folder_;
    QString resolved_path_;

    enum class State {
        Empty,
        Loading,
        Loaded,
        NotFound,
        CouldNotRead
    };

    State state_ = State::Empty;
    QImage loaded_image_;
    QPixmap cached_pixmap_;
    double playhead_ = -1.0;
    int gain_ = 1;

    // Static LRU cache for 64 images
    static constexpr size_t kMaxCacheSize = 64;
    static std::map<QString, QImage> s_cache_map;
    static std::list<QString> s_cache_lru;
    static std::mutex s_cache_mutex;
};

} // namespace whalemix::gui
