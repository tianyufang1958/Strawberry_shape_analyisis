cmake_minimum_required(VERSION 3.0.0)

project(PT_uniform VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Core)

link_directories(/usr/local/lib)
find_package(PCL 1.9 REQUIRED COMPONENTS common io io_ply visualization features filters surface)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
include(${VTK_USE_FILE})
find_package(VTK REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem system thread)

add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} Qt5::Core Boost::system Boost::filesystem ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${OpenCV_LIBS})

