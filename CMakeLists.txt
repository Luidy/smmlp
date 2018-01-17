cmake_minimum_required(VERSION 3.8)
project(smmlp)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp network.cpp network.h node.cpp node.h layer.cpp layer.h Sample.h utils.h)
add_executable(smmlp ${SOURCE_FILES})