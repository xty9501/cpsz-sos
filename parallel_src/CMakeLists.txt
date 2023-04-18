add_executable (cpsz_parallel cpsz_parallel.cpp)
target_include_directories (cpsz_parallel PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries (cpsz_parallel cpsz FTK::libftk zstd::libzstd_shared)

install (TARGETS cpsz_parallel
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)