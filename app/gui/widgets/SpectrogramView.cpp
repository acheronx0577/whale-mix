#include "SpectrogramView.h"
#include "Theme.h"
#include "Fonts.h"
#include <QPainter>
#include <QPainterPath>
#include <QFileInfo>
#include <QDir>
#include <QImageReader>
#include <QtConcurrent>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QKeyEvent>

namespace whalemix::gui {

std::map<QString, QImage> SpectrogramView::s_cache_map;
std::list<QString> SpectrogramView::s_cache_lru;
std::mutex SpectrogramView::s_cache_mutex;

SpectrogramView::SpectrogramView(QWidget* parent) : QWidget(parent) {
    setFixedHeight(130);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void SpectrogramView::set_playhead(double fraction) {
    if (std::abs(playhead_ - fraction) > 0.0005) {
        playhead_ = fraction;
        update();
    }
}

void SpectrogramView::set_gain(int gain) {
    if (gain_ != gain) {
        gain_ = gain;
        update();
    }
}

void SpectrogramView::clear() {
    image_file_.clear();
    resolved_path_.clear();
    loaded_image_ = QImage();
    cached_pixmap_ = QPixmap();
    state_ = State::Empty;
    update();
}

QString SpectrogramView::resolve_image_path(const QString& image_file, const QString& csv_folder) const {
    if (image_file.isEmpty()) return "";

    QFileInfo abs_fi(image_file);
    if (abs_fi.isAbsolute() && abs_fi.exists()) {
        return abs_fi.absoluteFilePath();
    }

    if (csv_folder.isEmpty()) return "";

    QDir dir(csv_folder);
    if (dir.exists(image_file)) {
        return dir.filePath(image_file);
    }
    if (dir.exists("images/" + image_file)) {
        return dir.filePath("images/" + image_file);
    }
    if (dir.exists("pictures/" + image_file)) {
        return dir.filePath("pictures/" + image_file);
    }

    return "";
}

void SpectrogramView::set_image_info(const QString& image_file, const QString& csv_folder) {
    image_file_ = image_file;
    csv_folder_ = csv_folder;

    if (image_file.isEmpty()) {
        state_ = State::Empty;
        loaded_image_ = QImage();
        resolved_path_.clear();
        update();
        return;
    }

    QString path = resolve_image_path(image_file, csv_folder);
    if (path.isEmpty()) {
        state_ = State::NotFound;
        loaded_image_ = QImage();
        resolved_path_.clear();
        update();
        return;
    }

    resolved_path_ = path;

    // Check LRU cache
    {
        std::lock_guard<std::mutex> lock(s_cache_mutex);
        auto it = s_cache_map.find(path);
        if (it != s_cache_map.end()) {
            loaded_image_ = it->second;
            cached_pixmap_ = QPixmap::fromImage(loaded_image_);
            state_ = State::Loaded;
            // Move key to front of LRU
            s_cache_lru.remove(path);
            s_cache_lru.push_front(path);
            update();
            return;
        }
    }

    // Load async
    load_image_async(path);
}

void SpectrogramView::load_image_async(const QString& resolved_path) {
    state_ = State::Loading;
    update();

    auto watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher, resolved_path]() {
        QImage img = watcher->result();
        watcher->deleteLater();

        if (resolved_path_ != resolved_path) {
            return; // Stale request
        }

        if (img.isNull()) {
            state_ = State::CouldNotRead;
            loaded_image_ = QImage();
            cached_pixmap_ = QPixmap();
        } else {
            state_ = State::Loaded;
            loaded_image_ = img;
            cached_pixmap_ = QPixmap::fromImage(loaded_image_);

            // Cache it
            std::lock_guard<std::mutex> lock(s_cache_mutex);
            if (s_cache_map.size() >= kMaxCacheSize) {
                QString oldest = s_cache_lru.back();
                s_cache_lru.pop_back();
                s_cache_map.erase(oldest);
            }
            s_cache_map[resolved_path] = img;
            s_cache_lru.push_front(resolved_path);
        }
        update();
    });

    QFuture<QImage> future = QtConcurrent::run([resolved_path]() {
        QImageReader reader(resolved_path);
        reader.setAutoTransform(true);
        return reader.read();
    });
    watcher->setFuture(future);
}

void SpectrogramView::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    int w = width();
    int h = height();

    // Box: height 130, padding 12, 1px dashed #c9d9f3 border, radius 8, mist fill
    QRectF box_rect(0.5, 0.5, w - 1.0, h - 1.0);
    p.setPen(Qt::NoPen);
    p.setBrush(theme::kMist);
    p.drawRoundedRect(box_rect, 8.0, 8.0);

    // 12px padded area inside
    QRectF pad_rect(12.0, 12.0, w - 24.0, h - 24.0);

    if (state_ == State::Loaded && !cached_pixmap_.isNull()) {
        QPainterPath clip;
        clip.addRoundedRect(pad_rect, 6.0, 6.0);
        p.save();
        p.setClipPath(clip);

        // Aspect-fill and crop vertically centered
        qreal target_w = pad_rect.width();
        qreal target_h = pad_rect.height();
        qreal img_w = cached_pixmap_.width();
        qreal img_h = cached_pixmap_.height();

        qreal scale = qMax(target_w / img_w, target_h / img_h);
        qreal draw_w = img_w * scale;
        qreal draw_h = img_h * scale;
        qreal draw_x = pad_rect.x() + (target_w - draw_w) / 2.0;
        qreal draw_y = pad_rect.y() + (target_h - draw_h) / 2.0;

        p.drawPixmap(QRectF(draw_x, draw_y, draw_w, draw_h), cached_pixmap_, cached_pixmap_.rect());

        if (gain_ > 1) {
            p.setCompositionMode(QPainter::CompositionMode_Screen);
            int alpha = (gain_ == 2) ? 35 : ((gain_ == 4) ? 70 : 110);
            p.fillRect(pad_rect, QColor(56, 189, 248, alpha));
            p.setCompositionMode(QPainter::CompositionMode_SourceOver);
        }

        // Draw traveling playhead across pad_rect
        if (playhead_ >= 0.0 && playhead_ <= 1.0) {
            int px = pad_rect.x() + static_cast<int>(playhead_ * pad_rect.width());
            QColor ph_color = (gain_ >= 8) ? QColor(56, 189, 248) : theme::kDeep;
            int line_w = (gain_ >= 4) ? 3 : 2;

            if (gain_ >= 2) {
                p.setPen(QPen(QColor(ph_color.red(), ph_color.green(), ph_color.blue(), 60), line_w + 4));
                p.drawLine(px, pad_rect.top(), px, pad_rect.bottom());
            }

            p.setPen(QPen(ph_color, line_w));
            p.drawLine(px, pad_rect.top(), px, pad_rect.bottom());
        }

        p.restore();
    }

    // Dashed border: 1px dashed #c9d9f3, radius 8
    QPen dash_pen(theme::kSpectroDashedBorder, 1.0, Qt::DashLine);
    p.setPen(dash_pen);
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(box_rect, 8.0, 8.0);

    // Caption: mono 13 slate on white 85% rounded backing (padding 4/8, radius 4)
    QString cap_text;
    switch (state_) {
    case State::Empty:
        cap_text = "image_file: \xe2\x80\x94"; // em dash
        break;
    case State::NotFound:
        cap_text = QString("image_file: %1 (not found)").arg(image_file_);
        break;
    case State::CouldNotRead:
        cap_text = QString("image_file: %1 (could not read)").arg(image_file_);
        break;
    case State::Loading:
    case State::Loaded:
        cap_text = QString("image_file: %1").arg(image_file_);
        break;
    }

    QFont cap_font = fonts::mono(13, 450);
    QFontMetrics fm(cap_font);
    int text_w = fm.horizontalAdvance(cap_text);
    int text_h = fm.height();

    // Backing: padding 4 top/bottom, 8 left/right
    int back_w = text_w + 16;
    int back_h = text_h + 8;
    QRectF back_rect(pad_rect.x() + 8, pad_rect.y() + 8, back_w, back_h);

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 255, 255, 217)); // 85% opacity white
    p.drawRoundedRect(back_rect, 4.0, 4.0);

    p.setFont(cap_font);
    p.setPen(theme::kSlate);
    int ty = back_rect.y() + 4 + fm.ascent();
    p.drawText(back_rect.x() + 8, ty, cap_text);
}

void SpectrogramView::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && state_ == State::Loaded && !loaded_image_.isNull()) {
        auto* dlg = new QDialog(this);
        dlg->setWindowTitle(QString("Spectrogram — %1").arg(image_file_));
        dlg->setModal(true);
        dlg->resize(qMin(1200, loaded_image_.width() + 40), qMin(800, loaded_image_.height() + 40));

        auto* layout = new QVBoxLayout(dlg);
        layout->setContentsMargins(16, 16, 16, 16);

        auto* scroll = new QScrollArea(dlg);
        scroll->setWidgetResizable(true);

        auto* label = new QLabel(scroll);
        label->setPixmap(QPixmap::fromImage(loaded_image_));
        label->setAlignment(Qt::AlignCenter);
        scroll->setWidget(label);

        layout->addWidget(scroll);
        dlg->show();
    }
}

} // namespace whalemix::gui
