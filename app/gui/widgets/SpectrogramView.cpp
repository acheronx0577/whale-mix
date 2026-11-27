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
#include <QFile>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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
    audio_caption_.clear();
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
    audio_caption_.clear();
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

void SpectrogramView::load_async(const QString& resolved_path, std::function<QImage()> load) {
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

    QFuture<QImage> future = QtConcurrent::run(std::move(load));
    watcher->setFuture(future);
}

void SpectrogramView::load_image_async(const QString& resolved_path) {
    load_async(resolved_path, [resolved_path]() {
        QImageReader reader(resolved_path);
        reader.setAutoTransform(true);
        return reader.read();
    });
}

void SpectrogramView::set_audio_window(const QString& wav_path, double start_s, double end_s, int channel) {
    const QString key = QString("wav:%1|%2|%3|%4").arg(wav_path).arg(start_s, 0, 'f', 3).arg(end_s, 0, 'f', 3).arg(channel);
    image_file_ = key;
    audio_caption_ = QString("%1 \xc2\xb7 ch%2 \xc2\xb7 %3\xe2\x80\x93%4 s \xc2\xb7 0\xe2\x80\x93" "250 Hz")
                         .arg(QFileInfo(wav_path).fileName()).arg(channel).arg(start_s, 0, 'f', 1).arg(end_s, 0, 'f', 1);
    csv_folder_.clear();
    resolved_path_ = key;
    {
        std::lock_guard<std::mutex> lock(s_cache_mutex);
        auto it = s_cache_map.find(key);
        if (it != s_cache_map.end()) {
            loaded_image_ = it->second;
            cached_pixmap_ = QPixmap::fromImage(loaded_image_);
            state_ = State::Loaded;
            s_cache_lru.remove(key);
            s_cache_lru.push_front(key);
            update();
            return;
        }
    }
    load_async(key, [wav_path, start_s, end_s, channel]() { return render_wav_window(wav_path, start_s, end_s, channel); });
}

namespace {
quint32 le32(const QByteArray& b, qsizetype at) {
    return quint32(quint8(b[at])) | quint32(quint8(b[at + 1])) << 8 | quint32(quint8(b[at + 2])) << 16 |
           quint32(quint8(b[at + 3])) << 24;
}
quint16 le16(const QByteArray& b, qsizetype at) { return quint16(quint8(b[at]) | quint8(b[at + 1]) << 8); }
}  // namespace

QImage SpectrogramView::render_wav_window(const QString& wav_path, double start_s, double end_s, int channel) {
    QFile f(wav_path);
    if (!f.open(QIODevice::ReadOnly) || end_s <= start_s) return {};
    const QByteArray head = f.read(12);
    if (head.size() < 12 || !head.startsWith("RIFF") || head.mid(8, 4) != "WAVE") return {};
    quint16 format = 0, channels = 0, bits = 0;
    quint32 rate = 0;
    qint64 data_pos = -1, data_len = 0;
    while (!f.atEnd()) {
        const QByteArray ch = f.read(8);
        if (ch.size() < 8) break;
        const quint32 len = le32(ch, 4);
        if (ch.startsWith("fmt ")) {
            const QByteArray fmt = f.read(len);
            if (fmt.size() < 16) return {};
            format = le16(fmt, 0);
            channels = le16(fmt, 2);
            rate = le32(fmt, 4);
            bits = le16(fmt, 14);
            if (format == 0xFFFE && fmt.size() >= 26) format = le16(fmt, 24);
            if (len & 1) f.read(1);
        } else if (ch.startsWith("data")) {
            data_pos = f.pos();
            data_len = len;
            break;
        } else if (!f.seek(f.pos() + len + (len & 1))) {
            break;
        }
    }
    const bool pcm16 = format == 1 && bits == 16, f32 = format == 3 && bits == 32;
    if (data_pos < 0 || rate == 0 || channels == 0 || channel >= channels || !(pcm16 || f32)) return {};
    const int bps = bits / 8, frame_bytes = bps * channels;
    const qint64 total = data_len / frame_bytes;
    const qint64 a = qBound<qint64>(0, qint64(start_s * rate), total);
    const qint64 b = qBound<qint64>(a, qint64(end_s * rate), total);
    if (b - a < 64 || !f.seek(data_pos + a * frame_bytes)) return {};
    const QByteArray raw = f.read((b - a) * frame_bytes);
    const qint64 n = raw.size() / frame_bytes;
    std::vector<double> x(static_cast<size_t>(n));
    for (qint64 i = 0; i < n; ++i) {
        const char* p = raw.constData() + i * frame_bytes + channel * bps;
        if (pcm16) {
            x[size_t(i)] = qint16(quint16(quint8(p[0]) | quint8(p[1]) << 8)) / 32768.0;
        } else {
            float v;
            std::memcpy(&v, p, 4);
            x[size_t(i)] = v;
        }
    }
    int nfft = 64;
    while (nfft < int(rate / 2) && nfft < 4096) nfft *= 2;  // ~0.5 s analysis window
    if (n < nfft) nfft = 64;
    const int hop = std::max(1, nfft / 8);
    const double fmax = std::min(250.0, rate / 2.0);
    const int bins = std::max(2, int(fmax / rate * nfft));
    const int frames = n >= nfft ? int((n - nfft) / hop) + 1 : 0;
    if (frames < 2) return {};
    std::vector<double> win(static_cast<size_t>(nfft));
    std::vector<double> db(static_cast<size_t>(frames) * static_cast<size_t>(bins));
    for (int i = 0; i < nfft; ++i) win[size_t(i)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (nfft - 1));
    std::vector<double> cos_t(static_cast<size_t>(nfft)), sin_t(static_cast<size_t>(nfft)), frame(static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        cos_t[size_t(i)] = std::cos(2.0 * M_PI * i / nfft);
        sin_t[size_t(i)] = std::sin(2.0 * M_PI * i / nfft);
    }
    for (int t = 0; t < frames; ++t) {
        const double* s = x.data() + size_t(t) * size_t(hop);
        for (int i = 0; i < nfft; ++i) frame[size_t(i)] = s[i] * win[size_t(i)];
        for (int k = 0; k < bins; ++k) {
            double re = 0, im = 0;
            size_t idx = 0;
            for (int i = 0; i < nfft; ++i) {
                re += frame[size_t(i)] * cos_t[idx];
                im -= frame[size_t(i)] * sin_t[idx];
                idx += size_t(k);
                if (idx >= size_t(nfft)) idx -= size_t(nfft);
            }
            db[size_t(t) * size_t(bins) + size_t(k)] = 10.0 * std::log10(re * re + im * im + 1e-20);
        }
    }
    std::vector<double> sorted = db;
    std::sort(sorted.begin(), sorted.end());
    const double lo = sorted[size_t(sorted.size() * 0.05)], hi = sorted[size_t((sorted.size() - 1) * 0.995)];
    QImage img(frames, bins, QImage::Format_RGB32);
    for (int t = 0; t < frames; ++t) {
        for (int k = 0; k < bins; ++k) {
            const double v = qBound(0.0, (db[size_t(t) * size_t(bins) + size_t(k)] - lo) / std::max(1e-9, hi - lo), 1.0);
            // paper (#f4f8ff) -> navy (#0b1640)
            const int r = int(0xf4 + (0x0b - 0xf4) * v), g = int(0xf8 + (0x16 - 0xf8) * v), bl = int(0xff + (0x40 - 0xff) * v);
            img.setPixel(t, bins - 1 - k, qRgb(r, g, bl));
        }
    }
    return img;
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
        cap_text = audio_caption_.isEmpty() ? QString("image_file: %1 (could not read)").arg(image_file_)
                                            : audio_caption_ + " (could not read audio)";
        break;
    case State::Loading:
    case State::Loaded:
        // No caption over a loaded picture: the inspector already shows the file and time right
        // above the spectrogram, and a banner here hid part of the call.
        return;
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
