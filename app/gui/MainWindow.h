#pragma once

#include <QMainWindow>
#include "AppState.h"
#include "widgets/TopBar.h"
#include "widgets/StatusBar.h"
#include "widgets/Sidebar.h"
#include "widgets/CenterPanel.h"
#include "widgets/DetailPanel.h"

namespace whalemix {

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

    AppState* state() { return &state_; }
    gui::TopBar* top_bar() { return top_bar_; }
    gui::StatusBar* status_bar() { return status_bar_; }
    gui::Sidebar* sidebar() { return sidebar_; }
    gui::CenterPanel* center_panel() { return center_panel_; }
    gui::DetailPanel* detail_panel() { return detail_panel_; }

    void open_file(const QString& path);
    void export_report();
    void set_manual_label(const QString& label);
    void undo_override();

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;

private:
    void update_all_views();

    AppState state_;
    gui::TopBar* top_bar_ = nullptr;
    gui::StatusBar* status_bar_ = nullptr;
    gui::Sidebar* sidebar_ = nullptr;
    gui::CenterPanel* center_panel_ = nullptr;
    gui::DetailPanel* detail_panel_ = nullptr;

    struct UndoEntry {
        int row_index;
        std::string prev_label;
        std::string prev_reason;
    };
    std::vector<UndoEntry> undo_stack_;
};

} // namespace whalemix
