//
// Created by James Miller on 9/3/2025.
//

#pragma once

#ifdef M_LOGGER
#undef M_LOGGER
#endif

class Logger
{
public:
    enum Verbosity
    {
        Info,
        Warning,
        Error,
        Debug
    };
    enum Category
    {
        LogTemp,
        LogGraphics,
        LogAudio,
        LogPhysics,
        LogCore,
        LogNetwork,
    };

    static void Log(Category category, Verbosity verbosity, const std::string& message, const char* file, int line)
    {
        const char* categoryStr = "";
        const char* verbosityStr = "";

        switch (category)
        {
            case LogTemp: categoryStr = "Temp"; break;
            case LogGraphics: categoryStr = "Graphics"; break;
            case LogAudio: categoryStr = "Audio"; break;
            case LogPhysics: categoryStr = "Physics"; break;
            case LogCore: categoryStr = "Core"; break;
            case LogNetwork: categoryStr = "Network"; break;
        }

        switch (verbosity)
        {
            case Info: verbosityStr = "INFO"; break;
            case Warning: verbosityStr = "WARNING"; break;
            case Error: verbosityStr = "ERROR"; break;
            case Debug: verbosityStr = "DEBUG"; break;
        }

        // Simple console output, can be replaced with file logging or other mechanisms
        printf("[%s] [%s] (%s:%d): %s\n", categoryStr, verbosityStr, file, line, message.c_str());
    }
};

#if ENABLE_DEBUG_LOGGING
#define M_LOGGER(category, verbosity, message) Logger::Log(category, verbosity, message, __FILE__, __LINE__)
#else
#define M_LOGGER(category, verbosity, message) ;
#endif