//
// Created by James Miller on 11/14/2025.
//

#pragma once
#include <chrono>

class DeltaTimer final
{
private:
    using clock = std::chrono::steady_clock;

    void Init()
    {
        last_ = clock::now();
    }

    // Returns delta time in seconds (double). Updates internal last time.
    double Tick() {
        auto now = clock::now();
        std::chrono::duration<double> dt = now - last_;
        last_ = now;
        return dt.count();
    }

    clock::time_point last_;
public:
    friend class Environment;
};