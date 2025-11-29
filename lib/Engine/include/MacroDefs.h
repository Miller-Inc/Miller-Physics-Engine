//
// Created by James Miller on 10/14/2025.
//

/// EngineTypes/MacroDefs.h
/// This file contains various macro definitions used throughout the engine,
///     including constants for maximum string lengths and other utility macros.
///     This is where some of the core game-independent definitions are stored.

#pragma once

/// Maximum length for string messages
#define MAX_STRING_MESSAGE_LEN 1024


#if defined(_DEBUG)
    /// Enable debug logging in debug builds
    #define ENABLE_DEBUG_LOGGING true
    /// Debug mode flag
    #define DEBUG_MODE true
#elif defined(NDEBUG)
    /// Disable debug logging in release builds
    #define ENABLE_DEBUG_LOGGING false
    /// Debug mode flag
    #define DEBUG_MODE false
#else
    /// Default to enabling debug logging if build type is unknown
    #define ENABLE_DEBUG_LOGGING true
    /// Debug mode flag
    #define DEBUG_MODE true
#endif


/// Platform detection macros (make sure only one is true, though all are defined)
#ifdef _WIN32
    /// Windows platform
    #define PLATFORM_WINDOWS true
    /// Unix-like platform (Linux, BSD, etc.)
    #define PLATFORM_UNIX false
    /// macOS platform
    #define PLATFORM_MACOS false
    /// Unknown platform
    #define PLATFORM_UNKNOWN false
    /// Determines if the platform is officially supported
    #define SUPPORTED true
#elif defined(__linux__) || defined(__unix__)
    /// Unix-like platform (Linux, BSD, etc.)
    #define PLATFORM_UNIX true
    /// Windows platform
    #define PLATFORM_WINDOWS false
    /// macOS platform
    #define PLATFORM_MACOS false
    /// Unknown platform
    #define PLATFORM_UNKNOWN false
    /// Determines if the platform is officially supported
    #define SUPPORTED true
#elif defined(__APPLE__)
    /// macOS platform
    #define PLATFORM_MACOS true
    /// Windows platform
    #define PLATFORM_WINDOWS false
    /// Unix-like platform (Linux, BSD, etc.)
    #define PLATFORM_UNIX false
    #define PLATFORM_UNKNOWN false
    /// Determines if the platform is officially supported
    #define SUPPORTED false
#else
    /// Unknown platform
    #define PLATFORM_UNKNOWN true
    /// Windows platform
    #define PLATFORM_WINDOWS false
    /// Unix-like platform (Linux, BSD, etc.)
    #define PLATFORM_UNIX false
    /// macOS platform
    #define PLATFORM_MACOS false
    /// Determines if the platform is officially supported
    #define SUPPORTED false
#endif

/// Flag to enable building on unsupported platforms (use with caution)
#define BUILD_NOT_SUPPORTED_PLATFORMS false


#if !SUPPORTED && !BUILD_NOT_SUPPORTED_PLATFORMS
    #error "Unsupported platform! This engine offically supports Windows or Linux. If you would like to try building on another platform, you may attempt to do so by changing the BUILD_NOT_SUPPORTED_PLATFORMS flag in the EngineTypes/MacroDefs.h file to true, but be aware that this is not guaranteed to work and you may run into issues."
#endif