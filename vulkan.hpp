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
#include "simple_render_system.hpp"
#include "cuda_render_system.hpp"

namespace v{

class Vulkan {
public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    Vulkan();
    ~Vulkan();

    Vulkan(const Vulkan&) = delete;
    Vulkan& operator=(const Vulkan&) = delete;

    void run();
private:
    void loadGameObjects();

    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    V_Renderer v_renderer{ v_window, v_device };

    std::vector<V_GameObject> gameObjects;

};
}
#endif
