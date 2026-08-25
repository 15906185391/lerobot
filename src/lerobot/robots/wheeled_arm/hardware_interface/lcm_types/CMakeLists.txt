cmake_minimum_required(VERSION 3.5)
project(biped_lcm_types)

set(INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})

find_package(lcm CONFIG REQUIRED)
include(${LCM_USE_FILE})
include(GNUInstallDirs)

file(GLOB_RECURSE lcm_sources
        LIST_DIRECTORIES false
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.lcm)

lcm_wrap_types(
        C_EXPORT my_lcmtypes
        C_SOURCES c_sources
        C_HEADERS c_headers
        CPP_HEADERS cpp_headers
        DESTINATION ${INCLUDE_DIRS}/cpp
        ${lcm_sources}
)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE lcm glib-2.0)
target_include_directories(
        ${PROJECT_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

lcm_add_library(_${PROJECT_NAME} CPP ${cpp_headers})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/cpp)
add_custom_target(lcm_build_source ALL
        COMMAND ${CMAKE_COMMAND} -E
        DEPENDS _${PROJECT_NAME}.sources
        )

install(DIRECTORY ${INCLUDE_DIRS}
        DESTINATION include
        )

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Config.cmake
        #CONFIGURATIONS Release
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
