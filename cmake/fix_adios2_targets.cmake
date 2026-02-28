#------------------------------------------------------------------------------#
# Fix ADIOS2 CMake target files
# This script generates the missing configuration-specific target files
# (e.g., adios2-cxx11-targets-Debug.cmake) that are needed for proper
# CMake target import when ADIOS2 is built in Debug mode.
#------------------------------------------------------------------------------#

set(ADIOS2_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/lib/cmake/adios2")
set(ADIOS2_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# Function to generate a configuration-specific target file
function(generate_config_target_file config_file config_name)
  if(NOT EXISTS "${config_file}")
    message(STATUS "Generating missing ADIOS2 target file: ${config_file}")
    
    # Determine which targets this file should contain and find actual library files
    if(config_file MATCHES "cxx11")
      set(targets "adios2::cxx11" "adios2::cxx11_mpi")
      set(lib_names "adios2_cxx11" "adios2_cxx11_mpi")
    else()
      set(targets "adios2::core" "adios2::core_mpi" "adios2::perfstubs")
      set(lib_names "adios2_core" "adios2_core_mpi" "adios2_perfstubs")
    endif()
    
    # Find actual library files (look for the full versioned .so file, not symlinks)
    set(libs "")
    set(sonames "")
    foreach(lib_name ${lib_names})
      # Look for the full versioned library file (e.g., libadios2_cxx11.so.2.10.1)
      file(GLOB lib_files "${ADIOS2_LIB_DIR}/lib${lib_name}.so.*.*.*")
      if(NOT lib_files)
        # Try with fewer version components
        file(GLOB lib_files "${ADIOS2_LIB_DIR}/lib${lib_name}.so.*.*")
      endif()
      if(NOT lib_files)
        # Try with just one version component
        file(GLOB lib_files "${ADIOS2_LIB_DIR}/lib${lib_name}.so.*")
      endif()
      
      if(lib_files)
        # Get the first match and extract just the filename
        list(GET lib_files 0 full_lib_path)
        get_filename_component(full_lib "${full_lib_path}" NAME)
        
        # Get the soname by removing the last version component (e.g., .so.2.10.1 -> .so.2.10)
        string(REGEX REPLACE "\\.[0-9]+$" "" soname "${full_lib}")
        
        list(APPEND libs "${full_lib}")
        list(APPEND sonames "${soname}")
      else()
        message(WARNING "Could not find library for ${lib_name}, skipping...")
        list(APPEND libs "")
        list(APPEND sonames "")
      endif()
    endforeach()
    
    # Generate the file content
    file(WRITE "${config_file}" "# Generated CMake target import file for ${config_name} configuration\n\n")
    
    list(LENGTH targets num_targets)
    math(EXPR num_targets "${num_targets} - 1")
    
    foreach(i RANGE ${num_targets})
      list(GET targets ${i} target)
      list(GET libs ${i} lib)
      list(GET sonames ${i} soname)
      
      if(lib AND soname)
        file(APPEND "${config_file}" 
          "# Import target \"${target}\" for configuration \"${config_name}\"\n"
          "set_property(TARGET ${target} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${config_name})\n"
          "set_target_properties(${target} PROPERTIES\n"
          "  IMPORTED_LOCATION_${config_name} \"\${_IMPORT_PREFIX}/lib/${lib}\"\n"
          "  IMPORTED_SONAME_${config_name} \"${soname}\"\n"
          "  )\n\n"
        )
      endif()
    endforeach()
    
    # Add import check targets
    file(APPEND "${config_file}" "list(APPEND _cmake_import_check_targets")
    foreach(i RANGE ${num_targets})
      list(GET targets ${i} target)
      list(GET libs ${i} lib)
      if(lib)
        file(APPEND "${config_file}" " ${target}")
      endif()
    endforeach()
    file(APPEND "${config_file}" " )\n")
    
    foreach(i RANGE ${num_targets})
      list(GET targets ${i} target)
      list(GET libs ${i} lib)
      if(lib)
        file(APPEND "${config_file}" 
          "list(APPEND _cmake_import_check_files_for_${target} \"\${_IMPORT_PREFIX}/lib/${lib}\" )\n"
        )
      endif()
    endforeach()
  endif()
endfunction()

# Check if ADIOS2 was installed
if(EXISTS "${ADIOS2_CMAKE_DIR}/adios2-cxx11-targets.cmake")
  message(STATUS "Fixing ADIOS2 CMake target files...")
  
  # Generate Debug and Release configuration files for cxx11 targets
  generate_config_target_file("${ADIOS2_CMAKE_DIR}/adios2-cxx11-targets-Debug.cmake" "Debug")
  generate_config_target_file("${ADIOS2_CMAKE_DIR}/adios2-cxx11-targets-Release.cmake" "Release")
  
  # Generate Debug and Release configuration files for core targets
  generate_config_target_file("${ADIOS2_CMAKE_DIR}/adios2-targets-Debug.cmake" "Debug")
  generate_config_target_file("${ADIOS2_CMAKE_DIR}/adios2-targets-Release.cmake" "Release")
  
  message(STATUS "ADIOS2 CMake target files fixed.")
else()
  message(STATUS "ADIOS2 not found in install directory, skipping target file generation.")
endif()
