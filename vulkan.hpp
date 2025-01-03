#ifndef VULKANCPP
#define VULKANCPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <memory>
#include <vector>

#include <cstdlib>
#include <iostream>

#include "v_window.hpp"
#include "v_device.hpp"
#include "v_gameobject.hpp"
#include "v_renderer.hpp"
#include "cuda_render_system.cuh"
#include "windowsSecurity.hpp"

namespace v{

class Vulkan {
public:
    static constexpr int WIDTH = 512*2;
    static constexpr int HEIGHT = 512;

    Vulkan();
    ~Vulkan();

    Vulkan(const Vulkan&) = delete;
    Vulkan& operator=(const Vulkan&) = delete;

    void run();
private:
    void loadGameObjects();
    void createSyncObjectsExt();

    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    V_Renderer v_renderer{ v_window, v_device };

    std::vector<V_GameObject> gameObjects;

    VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;

};
}
#endif
