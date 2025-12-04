#include <chrono>
#include <iostream>
#include <MillerCudaLibrary.h>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "GInstance.h"
#include "Windows/Viewport.h"

int main()
{
    std::cout << "Hello, World!" << std::endl;

    GInstance gameInstance{};

    Viewport viewport{};

    MWindow viewportWindow{
        "Viewport",
        [&viewport](const float deltaTime)
        {
            viewport.Draw(deltaTime);
        },
        [&viewport](GInstance* instance)
        {
            viewport.Init(instance);
            viewport.Open();
        },
        [&viewport](const float deltaTime)
        {
            viewport.Tick(deltaTime);
        },
        [&viewport]()
        {
            viewport.Close();
        },
        &viewport.isOpen
    };

    gameInstance.AddWindow(viewportWindow);

    gameInstance.Init();

    return 0;
}
