cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(real_time_monitoring VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Pylon REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(nlohmann_json REQUIRED)

# Add the Pylon SDK include directory
include_directories(/opt/pylon/include)

# Add the executable
add_executable(real_time_monitoring real_time_monitoring.cpp)

# Print the value of Pylon_INCLUDE_DIRS
message(STATUS "Pylon_INCLUDE_DIRS: ${Pylon_INCLUDE_DIRS}")

# Include directories
target_include_directories(real_time_monitoring PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(real_time_monitoring PRIVATE ${Pylon_INCLUDE_DIRS})
target_include_directories(real_time_monitoring PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(real_time_monitoring PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(real_time_monitoring PRIVATE ${nlohmann_json_INCLUDE_DIRS})

# Link libraries
target_link_libraries(real_time_monitoring PRIVATE ${OpenCV_LIBS})
target_link_libraries(real_time_monitoring PRIVATE ${Pylon_LIBRARIES})
target_link_libraries(real_time_monitoring PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(real_time_monitoring PRIVATE Boost::serialization)
target_link_libraries(real_time_monitoring PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(real_time_monitoring PRIVATE /opt/pylon/lib/libpylonbase.so)

# Add definitions for Torch
set_property(TARGET real_time_monitoring PROPERTY CXX_STANDARD 14)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  #target_compile_options(real_time_monitoring PRIVATE "-Wall" "-Wextra" "-Werror")
  target_compile_options(real_time_monitoring PRIVATE "-Wall" "-Wextra")
endif()

# Add WebSocket++ library
add_subdirectory(websocketpp)
target_link_libraries(real_time_monitoring PRIVATE websocketpp)