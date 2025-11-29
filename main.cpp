#include <chrono>
#include <iostream>
#include <MillerCudaLibrary.h>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "GInstance.h"

int main()
{
    std::cout << "Hello, World!" << std::endl;

    Environment env{};
    env.Init();
    env.GetDeltaTime();

    // Spawn the object and validate the returned pointer
    PhysicsObject* obj1 = env.SpawnPhysicsObject<PhysicsObject>();
    if (!obj1) {
        std::fprintf(stderr, "ERROR: SpawnPhysicsObject returned null\n");
        std::abort();
    }

    const Vector points[] = {
        Vector(0.0f, 0.0f, 0.0f),
        Vector(1.0f, 0.0f, 0.0f),
        Vector(0.0f, 1.0f, 0.0f)
    };

    // Extra guard before touching the object
    obj1->SetPoints(points, 3);

    std::printf("Object 1: %lld\n", (long long)obj1->GetIdentifier());

    for (int i = 0; i < 100; i++)
    {
        env.TickAll((float)env.GetDeltaTime());
    }

    GInstance gameInstance{};
    gameInstance.Init();

    return 0;
}
