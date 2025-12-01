//
// Created by James Miller on 11/14/2025.
//

#pragma once

/// Forward declaration of PhysicsObject
class PhysicsObject;

template<typename T>
struct Constructor
{
    using BeginPlayFncType = void(T::*)();
    using TickFncType = void(T::*)(float, PhysicsObject**, int);

    BeginPlayFncType BeginPlayFunc; // Function pointer to the BeginPlay method
    TickFncType TickFunc; // Function pointer to the Tick method

    Constructor() = default;
    Constructor(const BeginPlayFncType beginPlayFunc, const TickFncType tickFunc) :
        BeginPlayFunc(beginPlayFunc), TickFunc(tickFunc)
    {};
};
