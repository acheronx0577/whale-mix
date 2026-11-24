#include <QApplication>
#include <QGuiApplication>
#include <QTimer>
#include <QPixmap>
#include <iostream>
#include "MainWindow.h"
#include "Fonts.h"

int main(int argc, char* argv[]) {
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    // If running with --screenshot in CI or offscreen, ensure offscreen platform if no display
    bool has_screenshot = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--screenshot") {
            has_screenshot = true;
            break;
        }
    }

    QApplication app(argc, argv);
    whalemix::fonts::init();
    app.setFont(whalemix::fonts::body(14));

    whalemix::MainWindow window;
    window.resize(1440, 900);

    QString screenshot_path;
    QString open_path;
    QString select_id;

    for (int i = 1; i < argc; ++i) {
        QString arg = argv[i];
        if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg == "--open" && i + 1 < argc) {
            open_path = argv[++i];
        } else if (arg == "--select" && i + 1 < argc) {
            select_id = argv[++i];
        } else if (!arg.startsWith("--") && open_path.isEmpty()) {
            open_path = arg;
        }
    }

    if (!open_path.isEmpty()) {
        window.open_file(open_path);
    }

    if (!select_id.isEmpty()) {
        const auto& rows = window.state()->rows();
        int found = -1;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].candidate_id == select_id.toStdString() ||
                rows[i].sample_id == select_id.toStdString()) {
                found = static_cast<int>(i);
                break;
            }
        }
        if (found >= 0) {
            window.center_panel()->table()->select_model_index(found);
        }
    }

    if (!screenshot_path.isEmpty()) {
        window.setFixedSize(1440, 900);
        window.show();
        app.processEvents();

        QPixmap shot(1440, 900);
        shot.fill(Qt::white);
        window.render(&shot);

        if (!shot.save(screenshot_path)) {
            return 1;
        }
        return 0;
    }

    window.show();
    return app.exec();
}
