target_sources(unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/run_unit_tests.cpp
)

add_subdirectory(unit_tests)
