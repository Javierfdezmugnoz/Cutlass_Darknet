cmake_minimum_required(VERSION 3.18)
project(Aux_constructor LANGUAGES CUDA CXX)

#${CMAKE_CURRENT_SOURCE_DIR}/Simt_19.cu ${CMAKE_CURRENT_SOURCE_DIR}/Simt_19.h

add_library(Aux_constructor "")

    
target_sources(Aux_constructor
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cutlass_mmm.cu
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/cutlass_mmm.h
  )

target_include_directories(Aux_constructor
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    /home/javifdez/aux_projects/Adding_library/cutlass/examples/common
    /home/javifdez/aux_projects/Adding_library/cutlass/include
    /home/javifdez/aux_projects/Adding_library/cutlass/tools/util/include
)

# target_link_libraries(Aux_constructor
# PRIVATE
#     MMM_protected
#     CUTLASS
#     cutlass_tools_util_includes
# )