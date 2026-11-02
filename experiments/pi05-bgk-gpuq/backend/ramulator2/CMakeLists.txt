# Minimal H16 experiment build; upstream LICENSE remains in this directory.
cmake_minimum_required(VERSION 3.18)
project(Ramulator VERSION 2.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
# Sources are bundled locally. No FetchContent or network fallback is used.
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NB_TEST OFF CACHE BOOL "" FORCE)
set(NB_CREATE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/fmt)
add_subdirectory(ext/yaml-cpp)
add_subdirectory(ext/nanobind)
add_library(ramulator SHARED
  src/ramulator/base/factory.cpp
  src/ramulator/base/logger.cpp
  src/ramulator/base/utils.cpp
  src/ramulator/base/config.cpp
  src/ramulator/base/request.cpp
  src/ramulator/dram/node.cpp
  src/ramulator/dram/device.cpp
  src/ramulator/dram/dram_spec.cpp
  src/ramulator/dram/impl/LPDDR5XPIM.cpp
  src/ramulator/frontend/impl/memory_trace/loadstore_trace.cpp
  src/ramulator/memory_system/impl/generic_dram_system.cpp
  src/ramulator/memory_system/channel_mapper/impl/cache_line_interleave.cpp
  src/ramulator/controller/controller_base.cpp
  src/ramulator/controller/impl/lpddr_controller_base.cpp
  src/ramulator/controller/impl/lpddr5_controller.cpp
  src/ramulator/controller/scheduler/impl/frfcfs_rowhit.cpp
  src/ramulator/controller/refresh/impl/all_bank.cpp
  src/ramulator/controller/refresh/impl/no_refresh.cpp
  src/ramulator/controller/rowpolicy/impl/open.cpp
  src/ramulator/controller/addr_mapper/addr_mapper_base.cpp
  src/ramulator/controller/addr_mapper/impl/ro_ba_ra_co_ch.cpp
  src/ramulator/controller/plugin/impl/command_counter.cpp
  src/ramulator/controller/plugin/impl/cmd_trace_recorder.cpp
)
target_include_directories(ramulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ramulator PRIVATE fmt::fmt yaml-cpp::yaml-cpp)
set_target_properties(ramulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# Preserve live Python DSL/codegen. Only LPDDR5XPIM's generated C++ is compiled;
# small complete source metadata/import chains remain for wrapper generation.
add_custom_target(codegen
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    "PYTHONDONTWRITEBYTECODE=1" "PYTHONNOUSERSITE=1"
    ${Python_EXECUTABLE} -m ramulator codegen LPDDR5XPIM
    --src-dir ${CMAKE_CURRENT_SOURCE_DIR}/src/ramulator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)
add_dependencies(ramulator codegen)
nanobind_add_module(_ramulator src/ramulator/python/bindings.cpp)
target_link_libraries(_ramulator PRIVATE ramulator)
target_include_directories(_ramulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(_ramulator PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ramulator)
