cmake_minimum_required(VERSION 3.25)

project(WhaleMix VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Treat warnings as errors when requested or in CI
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
if(WARNINGS_AS_ERRORS)
  if(MSVC)
    add_compile_options(/WX)
  else()
    add_compile_options(-Werror -Wall -Wextra)
  endif()
endif()

# Compiler performance optimizations
if(MSVC)
  add_compile_options(/O2 /Oi /Gy)
else()
  add_compile_options(-O3)
endif()

# Find Qt 6
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Widgets Svg Test Concurrent)
qt_standard_project_setup()
set(CMAKE_AUTORCC ON)

# Fix QTBUG-137687: Modern macOS SDK removed legacy AGL.framework
if(APPLE)
  if(TARGET WrapOpenGL::WrapOpenGL)
    get_target_property(_gl_libs WrapOpenGL::WrapOpenGL INTERFACE_LINK_LIBRARIES)
    if(_gl_libs)
      list(FILTER _gl_libs EXCLUDE REGEX "AGL")
      set_target_properties(WrapOpenGL::WrapOpenGL PROPERTIES INTERFACE_LINK_LIBRARIES "${_gl_libs}")
    endif()
  endif()
  if(TARGET Qt6::Gui)
    get_target_property(_gui_libs Qt6::Gui INTERFACE_LINK_LIBRARIES)
    if(_gui_libs)
      list(FILTER _gui_libs EXCLUDE REGEX "AGL")
      set_target_properties(Qt6::Gui PROPERTIES INTERFACE_LINK_LIBRARIES "${_gui_libs}")
    endif()
  endif()
endif()

# GoogleTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Core library (std C++ only, no Qt)
file(GLOB_RECURSE CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp
)
file(GLOB_RECURSE CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include/*.h
)

add_library(whalemix_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(whalemix_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)
target_compile_features(whalemix_core PUBLIC cxx_std_20)

# CLI executable
add_executable(whale-mix-cli
  cli/main.cpp
)
target_link_libraries(whale-mix-cli PRIVATE whalemix_core)
target_compile_features(whale-mix-cli PRIVATE cxx_std_20)

# GUI executable
set(GUI_LIB_SOURCES
  gui/MainWindow.cpp
  gui/EngineBridge.cpp
  gui/AppState.cpp
  gui/Fonts.cpp
  gui/widgets/Chip.cpp
  gui/widgets/Kbd.cpp
  gui/widgets/PillButton.cpp
  gui/widgets/Eyebrow.cpp
  gui/widgets/TopBar.cpp
  gui/widgets/StatusBar.cpp
  gui/widgets/Sidebar.cpp
  gui/widgets/SearchBox.cpp
  gui/widgets/ScoreTable.cpp
  gui/widgets/CenterPanel.cpp
  gui/widgets/SpectrogramView.cpp
  gui/widgets/DetailPanel.cpp
)
file(GLOB_RECURSE WIDGET_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gui/widgets/*.cpp
)
list(APPEND GUI_LIB_SOURCES ${WIDGET_SOURCES})
list(REMOVE_DUPLICATES GUI_LIB_SOURCES)

set(GUI_SOURCES
  gui/main.cpp
  ${GUI_LIB_SOURCES}
)

set(GUI_RESOURCES
  gui/resources/whale-mix.qrc
)

set(GUI_EXTRA_SOURCES "")
if(WIN32)
  set(GENERATED_RC "${CMAKE_CURRENT_BINARY_DIR}/whale-mix.rc")
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/gui/win/whale-mix.rc.in
    ${GENERATED_RC}
    @ONLY
  )
  list(APPEND GUI_EXTRA_SOURCES ${GENERATED_RC})
elseif(APPLE)
  set(MACOSX_BUNDLE_ICON_FILE whale-mix.icns)
  set(MACOSX_BUNDLE_GUI_IDENTIFIER org.acheronx.whale-mix)
  set(MACOSX_BUNDLE_BUNDLE_NAME "Whale Mix")
  set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION})
  set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION})
  set_source_files_properties(gui/mac/whale-mix.icns PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
  list(APPEND GUI_EXTRA_SOURCES gui/mac/whale-mix.icns)
endif()

qt_add_executable(whale-mix
  WIN32
  MACOSX_BUNDLE
  ${GUI_SOURCES}
  ${GUI_RESOURCES}
  ${GUI_EXTRA_SOURCES}
)

if(APPLE)
  set_target_properties(whale-mix PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/gui/mac/Info.plist.in
    OUTPUT_NAME "Whale Mix"
  )
else()
  set_target_properties(whale-mix PROPERTIES
    OUTPUT_NAME "whale-mix"
  )
endif()

target_include_directories(whale-mix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gui
  ${CMAKE_CURRENT_SOURCE_DIR}/gui/widgets
)

target_link_libraries(whale-mix PRIVATE
  whalemix_core
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Svg
  Qt6::Concurrent
)
if(WIN32)
  target_link_libraries(whale-mix PRIVATE dwmapi)
endif()

# Testing
enable_testing()

# Core & golden tests
add_executable(whalemix_tests
  tests/core_tests.cpp
  tests/golden_tests.cpp
)
target_compile_definitions(whalemix_tests PRIVATE
  WHALEMIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(whalemix_tests PRIVATE
  whalemix_core
  GTest::gtest_main
)
target_include_directories(whalemix_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
add_test(NAME whalemix_tests COMMAND whalemix_tests)

# CLI golden test
add_test(NAME cli_golden_test
  COMMAND ${CMAKE_COMMAND}
    -DCLI_EXE=$<TARGET_FILE:whale-mix-cli>
    -DSAMPLES_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../samples
    -DGOLDEN_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/golden
    -DTEST_TEMP_DIR=${CMAKE_BINARY_DIR}/temp_test_cli
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_tests.cmake
)

# GUI smoke tests
qt_add_executable(whalemix_gui_tests
  tests/gui_smoke_tests.cpp
  tests/engine_bridge_tests.cpp
  ${GUI_LIB_SOURCES}
  ${GUI_RESOURCES}
)
target_compile_definitions(whalemix_gui_tests PRIVATE
  WHALEMIX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_include_directories(whalemix_gui_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gui
  ${CMAKE_CURRENT_SOURCE_DIR}/gui/widgets
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_link_libraries(whalemix_gui_tests PRIVATE
  whalemix_core
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Svg
  Qt6::Concurrent
  Qt6::Test
  GTest::gtest
)
if(WIN32)
  target_link_libraries(whalemix_gui_tests PRIVATE dwmapi)
endif()
add_test(NAME whalemix_gui_tests COMMAND whalemix_gui_tests)
# Optional fixture-engine checks: point these at a locally built worker and an untrained fixture
# bundle exported outside Git (python -m whalemix_training export --fixture). Empty = skipped.
set(WHALEMIX_TEST_ENGINE "" CACHE FILEPATH "whale-mix-engine binary for fixture GUI tests")
set(WHALEMIX_TEST_FIXTURE_BUNDLE "" CACHE PATH "untrained fixture bundle for engine tests")
set(WHALEMIX_TEST_PARITY_WAV "" CACHE FILEPATH "WAV whose <bundle>-parity-reference.json was exported with the fixture")
# CI turns this on so fixture-dependent tests fail instead of skipping when a path is missing.
option(WHALEMIX_REQUIRE_FIXTURE_TESTS "Fail (not skip) fixture-dependent GUI and engine tests" OFF)
set(_fixture_env "")
if(WHALEMIX_REQUIRE_FIXTURE_TESTS)
  list(APPEND _fixture_env "WHALEMIX_REQUIRE_FIXTURE=1")
endif()
set(_gui_test_env "QT_QPA_PLATFORM=offscreen" ${_fixture_env})
if(WHALEMIX_TEST_ENGINE AND WHALEMIX_TEST_FIXTURE_BUNDLE)
  list(APPEND _gui_test_env "WHALEMIX_ENGINE=${WHALEMIX_TEST_ENGINE}"
                            "WHALEMIX_FIXTURE_BUNDLE=${WHALEMIX_TEST_FIXTURE_BUNDLE}")
endif()
set_tests_properties(whalemix_gui_tests PROPERTIES ENVIRONMENT "${_gui_test_env}")

# Optional Rust worker tests through CTest.
option(WHALEMIX_ENGINE_TESTS "Run cargo test for app/engine from CTest" OFF)
if(WHALEMIX_ENGINE_TESTS)
  find_program(CARGO_EXECUTABLE cargo REQUIRED)
  add_test(NAME engine_cargo_tests
    COMMAND ${CARGO_EXECUTABLE} test --release --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/engine/Cargo.toml)
  set(_engine_test_env ${_fixture_env})
  if(WHALEMIX_TEST_FIXTURE_BUNDLE)
    list(APPEND _engine_test_env "WHALEMIX_FIXTURE_BUNDLE=${WHALEMIX_TEST_FIXTURE_BUNDLE}")
  endif()
  if(WHALEMIX_TEST_PARITY_WAV)
    list(APPEND _engine_test_env "WHALEMIX_PARITY_WAV=${WHALEMIX_TEST_PARITY_WAV}")
  endif()
  if(_engine_test_env)
    set_tests_properties(engine_cargo_tests PROPERTIES ENVIRONMENT "${_engine_test_env}")
  endif()
endif()

# Installation for portable zip
install(TARGETS whale-mix whale-mix-cli
  RUNTIME DESTINATION .
  BUNDLE DESTINATION .
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../samples
  DESTINATION .
  OPTIONAL
)
