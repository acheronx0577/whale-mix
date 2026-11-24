#pragma once

#include <QFont>
#include <QString>
#include <QStringList>

namespace whalemix::fonts {

void init();

// Strictly Google Sans Flex and Google Sans Code only
const QString& sans_family();
const QString& mono_family();
const QStringList& sans_families();
const QStringList& mono_families();

// Primary typeface: strictly Google Sans Flex only
// Headings (H1–H6): font-weight 450, letter-spacing: -2% (-0.02em)
QFont heading(int pixel_size, int weight = 450, qreal letter_spacing_em = -0.02);

// Body (p): font-weight 400, letter-spacing: -2% (-0.02em)
QFont body(int pixel_size, int weight = 400, qreal letter_spacing_em = -0.02);

// General sans accessor: strictly Google Sans Flex only
QFont sans(int pixel_size, int weight = 400, qreal letter_spacing_em = -0.02);

// Monospace font: strictly Google Sans Code only
// Data readings, timestamps, scores, badges, and uppercase technical tags (p.mono)
// font-weight 450
QFont mono(int pixel_size, int weight = 450, qreal letter_spacing_em = 0.0);

} // namespace whalemix::fonts
