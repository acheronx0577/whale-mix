#include "Fonts.h"
#include <QFontDatabase>
#include <QStringList>
#include <QFile>
#include <QCoreApplication>
#include <QDir>
#include <unordered_map>
#include <cstdint>

namespace whalemix::fonts {

namespace {
const QString kSansFamily = "Google Sans Flex";
const QString kMonoFamily = "Google Sans Code";
QStringList g_sans_families = { kSansFamily };
QStringList g_mono_families = { kMonoFamily };
bool g_initialized = false;

struct FontKey {
    int family_type; // 0 = sans, 1 = mono
    int pixel_size;
    int weight;
    int spacing_int; // letter_spacing_em * 1000

    bool operator==(const FontKey& o) const {
        return family_type == o.family_type &&
               pixel_size == o.pixel_size &&
               weight == o.weight &&
               spacing_int == o.spacing_int;
    }
};

struct FontKeyHash {
    std::size_t operator()(const FontKey& k) const {
        return (static_cast<std::size_t>(k.family_type) << 28) ^
               (static_cast<std::size_t>(k.pixel_size) << 20) ^
               (static_cast<std::size_t>(k.weight) << 10) ^
               static_cast<std::size_t>(k.spacing_int);
    }
};

std::unordered_map<FontKey, QFont, FontKeyHash> g_font_cache;
}

void init() {
    if (g_initialized) return;
    g_initialized = true;

    // 1. Embedded Qt resource fonts (strictly Google Sans Flex and Google Sans Code)
    QStringList font_files = {
        ":/fonts/GoogleSansFlex-Regular.ttf",
        ":/fonts/GoogleSansFlex-Medium.ttf",
        ":/fonts/GoogleSansFlex-SemiBold.ttf",
        ":/fonts/GoogleSansFlex-Bold.ttf",
        ":/fonts/GoogleSansCode-Regular.ttf",
        ":/fonts/GoogleSansCode-Medium.ttf",
        ":/fonts/GoogleSansCode-SemiBold.ttf"
    };

    for (const auto& file : font_files) {
        QFontDatabase::addApplicationFont(file);
    }

    // 2. Relative disk fallback (if running tests or unbundled binary)
    QString app_dir = QCoreApplication::applicationDirPath();
    QStringList search_dirs = {
        app_dir + "/assets/fonts",
        app_dir + "/../app/assets/fonts",
        app_dir + "/../../app/assets/fonts",
        "D:/Code/Projects/Marine-Science/MBARI-Portfolio/Blue-Whale-Call-Detection-Under-Ocean-Noise/app/assets/fonts"
    };

    for (const auto& sdir : search_dirs) {
        if (QDir(sdir).exists()) {
            QDir dir(sdir);
            for (const auto& fi : dir.entryInfoList({"*.ttf"}, QDir::Files)) {
                QFontDatabase::addApplicationFont(fi.absoluteFilePath());
            }
        }
    }
}

const QString& sans_family() {
    init();
    return kSansFamily;
}

const QString& mono_family() {
    init();
    return kMonoFamily;
}

const QStringList& sans_families() {
    init();
    return g_sans_families;
}

const QStringList& mono_families() {
    init();
    return g_mono_families;
}

QFont heading(int pixel_size, int weight, qreal letter_spacing_em) {
    return sans(pixel_size, weight, letter_spacing_em);
}

QFont body(int pixel_size, int weight, qreal letter_spacing_em) {
    return sans(pixel_size, weight, letter_spacing_em);
}

QFont sans(int pixel_size, int weight, qreal letter_spacing_em) {
    init();
    FontKey key{0, pixel_size, weight, static_cast<int>(letter_spacing_em * 1000.0)};
    auto it = g_font_cache.find(key);
    if (it != g_font_cache.end()) {
        return it->second;
    }

    QFont f(kSansFamily);
    f.setFamilies(g_sans_families);
    f.setPixelSize(pixel_size);
    f.setWeight(static_cast<QFont::Weight>(weight));
    if (letter_spacing_em != 0.0) {
        f.setLetterSpacing(QFont::AbsoluteSpacing, letter_spacing_em * pixel_size);
    }
    g_font_cache[key] = f;
    return f;
}

QFont mono(int pixel_size, int weight, qreal letter_spacing_em) {
    init();
    FontKey key{1, pixel_size, weight, static_cast<int>(letter_spacing_em * 1000.0)};
    auto it = g_font_cache.find(key);
    if (it != g_font_cache.end()) {
        return it->second;
    }

    QFont f(kMonoFamily);
    f.setFamilies(g_mono_families);
    f.setPixelSize(pixel_size);
    f.setWeight(static_cast<QFont::Weight>(weight));
    if (letter_spacing_em != 0.0) {
        f.setLetterSpacing(QFont::AbsoluteSpacing, letter_spacing_em * pixel_size);
    }
    g_font_cache[key] = f;
    return f;
}

} // namespace whalemix::fonts
