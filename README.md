# Miller Physics Engine
**Hello There** 👋 Welcome to the Miller Physics Engine repository! This 
project is dedicated to developing a robust and efficient physics engine 
for simulating real-world physical interactions in computer graphics and 
game development. The Miller Physics Engine aims to provide developers with 
a powerful toolset for creating realistic simulations of rigid body dynamics, 
collision detection, and response, all while maintaining high performance and 
real-time capabilities. While this project is currently under development, we are 
excited to share our progress and invite contributions from the community. Stay 
tuned for updates as we continue to build and refine the Miller Physics Engine!

## Features
- CUDA-accelerated computations for high performance
- Object-oriented design for easy integration
- Similar to Unreal Engine's C++ API for familiarity
- Modular architecture for extensibility

> Note: This project is currently in the early stages of development 
> and has only been tested on Windows with NVIDIA GPUs. Future updates 
> will include support for additional platforms and hardware (e.g., Ubuntu, 
> RHEL, & Vulkan). Mac support is not planned at this time.

## Getting Started
To get started with the Miller Physics Engine, follow these steps:

> Note: This project is currently optimized for NVIDIA GPUs using CUDA. 
> Support for other platforms and APIs (such as OpenCL or Vulkan) may 
> be added in future releases. 

1. Ensure you have the following prerequisites installed:
   - C++17 compatible compiler
   - CMake 3.10 or higher
   - CUDA Toolkit (if using GPU acceleration which is currently the only option)

2. Clone the repository:
   ```bash
   # SSH
   git clone git@github.com:Miller-Inc/Miller-Physics-Engine.git --recurse-submodules
   
    # HTTPS
    git clone https://github.com/Miller-Inc/Miller-Physics-Engine.git
    ```
   
3. Navigate to the project directory:
   ```bash
   cd Miller-Physics-Engine
   ```

4. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
   
## Current Status & Progress
The Miller Physics Engine is currently in the early stages of development. The core architecture is being 
designed, and initial implementations of rigid body dynamics and collision detection are underway. 
We are actively working on optimizing performance and ensuring the engine can handle complex 
simulations efficiently.

![Recording 12-01-2025](Resources/Miller%20Inc%20Content%20-%20Physics%20Engine%20Preview.gif)