#include "vulkan.hpp"

namespace v {
void Vulkan::run() {
    SimpleRenderSystem simpleRenderSystem{ v_device, v_renderer.getSwapChainRenderPass() };

    while (!v_window.shouldClose()) {
        glfwPollEvents();

        if (auto commandBuffer = v_renderer.beginFrame()) {
            v_renderer.beginSwapChainRenderPass(commandBuffer);
            simpleRenderSystem.renderGameObjects(commandBuffer, gameObjects);
            v_renderer.endSwapChainRenderPass(commandBuffer);
            v_renderer.endFrame();
        }
    }
    
    vkDeviceWaitIdle(v_device.device());
}


Vulkan::Vulkan() {
    loadGameObjects();
}

Vulkan::~Vulkan() {}

void Vulkan::loadGameObjects() {
    std::vector<V_Model::Vertex> vertices {
        {{0.0f, -0.5f}, { 1.0f, 0, 1.0f }},
        { {0.5f, 0.5f}, {1.0f, 1.0f, 0} },
        { {-0.5, 0.5f},{0, 1.0f, 1.0f} }
    };
    auto v_model = std::make_shared<V_Model>(v_device, vertices);

    auto triangle = V_GameObject::createGameObject();
    triangle.model = v_model;
    triangle.color = { 0.f, 1.f, .5f };
    triangle.transform2d.scale = { 1.f, 1.f };
    triangle.transform2d.rotation = .25f * glm::two_pi<float>();

    gameObjects.push_back(std::move(triangle));
}
}