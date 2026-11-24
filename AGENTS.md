# Blue Whale Call Detection Under Ocean Noise — Agent Instructions

## Dual-Stack Repository Architecture

This repository contains two completely independent software stacks:

1. **`site/` — Next.js 16 Web Application**
   - Marketing, scientific documentation, research logs, and downloads.
   - Deploys automatically to Vercel production on push to `main`.
   - Toolchain: Node.js, `npm run dev`, `npm run build`.
   - Stack: Next.js 16 (App Router with `proxy.js`), CSS modules, GSAP 3 / SplitText, Lenis smooth scrolling.

2. **`app/` — Native C++20 / Qt 6.8 Desktop Client & CLI (`whale-mix`)**
   - Bioacoustic window triage, audio spectrogram visualization, Blue A / Laya label mixing.
   - Toolchain: CMake 3.28+, Visual Studio 2022 (MSVC) / Apple Clang, Qt 6.8.
   - Build Presets: `windows-msvc`, `windows-msvc-release`, `macos`, `macos-release`.
   - Tests: GoogleTest (`app/tests/gui_smoke_tests.cpp`).

---

## Strict Domain Scoping Rules (Prevent Unnecessary Full-Codebase Scans)

### 1. Website Tasks (UI, Styling, Copy, Routes, Animations, Vercel)
- **Scope:** Operate EXCLUSIVELY inside `site/`.
- **Zero C++ Scans:** Antigravity MUST NEVER scan, inspect, grep, or build C++ files in `app/`, `whale_mix/`, or `CMakeLists.txt` for website tasks.
- **Working Tree Isolation:** Pre-existing uncommitted edits in `app/gui/` must remain untouched and MUST NEVER be staged or committed with website PRs or pushes.
- **Verification Fast Path:** Run `npm --prefix site run build`. Never run CMake, CTest, or full-repo Sonar for website updates.
- **Production Verification:** Verify deployment status via `npx vercel ls` from `site/`.

### 2. Desktop App Tasks (C++, Qt 6.8, Spectrograms, GUI, CMake)
- **Scope:** Operate EXCLUSIVELY inside `app/`, `packaging/`, and root CMake files.
- **Key Modules:**
  - `app/gui/MainWindow.cpp`, `MainWindow.h` — Main window layout, window state, menu bar.
  - `app/gui/widgets/SpectrogramView.cpp`, `SpectrogramView.h` — Spectrogram rendering, timeline zoom, playback cursor.
  - `app/gui/widgets/DetailPanel.cpp`, `DetailPanel.h` — Window inspector, audio metadata, Laya decision breakdown.
  - `app/gui/widgets/TopBar.cpp`, `TopBar.h` — Source file picker, pile filters, triage action triggers.
  - `app/cli/` — Headless `whale-mix-cli` binary (`--scores`, `--out`, `--noise`).
- **Zero Web Scans:** Never run npm, inspect `site/`, or check Vercel during desktop tasks.
- **Verification Fast Path:**
  - Configure: `cmake --preset windows-msvc`
  - Build: `cmake --build --preset windows-msvc-release`
  - Test: `ctest --preset windows-msvc-release --output-on-failure` (runs GoogleTest smoke tests)

---

## Git & Quality Workflow
- Keep diffs minimal and targeted to the requested domain (`site/` or `app/`).
- Verify builds with 0 errors before staging.
- Never execute `git commit` without explicit user permission.
