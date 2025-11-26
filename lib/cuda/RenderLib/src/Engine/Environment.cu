//
// Created by James Miller on 11/14/2025.
//

#include "Engine/Environment.h"
#include <cuda_runtime.h>

void Environment::TickAll(float deltaTime)
{
    for (PhysicsObject* obj : PhysicsObjects)
    {
        if (obj)
        {
            obj->Tick(deltaTime);
        }
    }
    cudaDeviceSynchronize();
    fflush(stdout);
}
