#include <chrono>
#include <iostream>
#include <MillerCudaLibrary.h>
#include <thread>
#include <Engine/EngineClock.h>

int main()
{
    std::cout << "Hello, World!" << std::endl;

    Environment env{};
    env.Init();

    PhysicsObject* obj1 = env.SpawnPhysicsObject(PhysicsObject::StaticClass());
    const Vector points[] = {
        Vector(0.0f, 0.0f, 0.0f),
        Vector(1.0f, 0.0f, 0.0f),
        Vector(0.0f, 1.0f, 0.0f)
    };
    obj1->SetPoints(points, 3);

    printf("Object 1: %lld", obj1->GetIdentifier());
    env.Init();

    for (int i = 0; i < 100; i++)
    {
        env.TickAll((float)env.GetDeltaTime());
        // env.TickAll(0.05f); // Simulate a fixed delta time of 0.05 seconds
    }


    /// With two objects, fps: ~2500

    return 0;
}
