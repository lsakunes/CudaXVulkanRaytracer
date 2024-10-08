#ifndef VULKANCPP
#define VULKANCPP
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "v_window.hpp"
#include "v_pipeline.hpp"
#include "v_device.hpp"

namespace v{
class Vulkan {
public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    void run();
private:
    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    V_Pipeline v_pipeline{v_device,
        "./shaders/vert.spv",
        "./shaders/frag.spv",
        V_Pipeline::defaultPipelineConfigInfo(WIDTH, HEIGHT)};
};

void Vulkan::run() {
    while (!v_window.shouldClose()) {
        glfwPollEvents();
    }
}
}
#endif