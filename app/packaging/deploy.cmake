if(WIN32)
  find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt windeployqt6 windeployqt-qt6 HINTS ${QT_BIN_DIR} ENV QT_ROOT)
  if(WINDEPLOYQT_EXECUTABLE)
    message(STATUS "Running windeployqt: ${WINDEPLOYQT_EXECUTABLE}")
    execute_process(
      COMMAND ${WINDEPLOYQT_EXECUTABLE} --release --no-translations "${DIST_DIR}/whale-mix.exe"
      RESULT_VARIABLE res
    )
    if(res EQUAL 0)
      message(STATUS "windeployqt succeeded")
    else()
      message(WARNING "windeployqt exited with code ${res}")
    endif()
  else()
    message(STATUS "windeployqt not found")
  endif()
endif()
