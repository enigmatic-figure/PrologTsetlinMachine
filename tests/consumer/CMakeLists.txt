cmake_minimum_required(VERSION 3.20)
project(PTMCleanConsumer LANGUAGES C CXX)

find_package(PTM 0.1 CONFIG REQUIRED)

add_executable(ptm_consumer_smoke main.c)
target_link_libraries(ptm_consumer_smoke PRIVATE PTM::runtime)

enable_testing()
add_test(NAME ptm_consumer_smoke COMMAND ptm_consumer_smoke)
