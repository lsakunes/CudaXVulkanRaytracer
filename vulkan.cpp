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
std::unique_ptr<V_Model> createCubeModel(V_Device& device, glm::vec3 offset) {
    std::vector<V_Model::Vertex> vertices{

        // left face (white)
        {{-.5f, -.5f, -.5f}, { .9f, .9f, .9f }}, //white
        { {-.5f, .5f, .5f},  {.1f, .8f, .8f} }, //cyan
        { {-.5f, -.5f, .5f}, {.1f, .1f, .8f} }, //blue
        { {-.5f, -.5f, -.5f}, { .9f, .9f, .9f } }, //white
        { {-.5f, .5f, -.5f}, {.8f, .1f, .1f} }, //red
        { {-.5f, .5f, .5f},  {.1f, .8f, .8f} }, //cyan

            // right face (yellow)
        { {.5f, -.5f, -.5f}, {.8f, .8f, .1f} }, //yellow
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta
        { {.5f, -.5f, .5f}, {.9f, .6f, .1f} }, //orange
        { {.5f, -.5f, -.5f}, {.8f, .8f, .1f} }, //yellow
        { {.5f, .5f, -.5f}, {.1f, .8f, .1f} }, //green
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta

        // top face (orange, remember y axis points down)

        { {-.5f, -.5f, -.5f}, { .9f, .9f, .9f } }, //white
        { {.5f, -.5f, .5f}, {.9f, .6f, .1f} }, //orange
        { {-.5f, -.5f, .5f}, {.1f, .1f, .8f} }, //blue
        { {-.5f, -.5f, -.5f}, { .9f, .9f, .9f } }, //white
        { {.5f, -.5f, -.5f}, {.8f, .8f, .1f} }, //yellow
        { {.5f, -.5f, .5f}, {.9f, .6f, .1f} }, //orange

            // bottom face (red)
        { {-.5f, .5f, -.5f}, {.8f, .1f, .1f} }, //red
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta
        { {-.5f, .5f, .5f}, {.1f, .8f, .8f} }, //cyan
        { {-.5f, .5f, -.5f}, {.8f, .1f, .1f} }, //red
        { {.5f, .5f, -.5f}, {.1f, .8f, .1f} }, //green
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta

            // nose face (blue)
        { {-.5f, -.5f, .5f}, {.1f, .1f, .8f} }, //blue
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta
        { {-.5f, .5f, .5f}, {.1f, .8f, .8f} }, //cyan
        { {-.5f, -.5f, .5f}, {.1f, .1f, .8f} }, //blue
        { {.5f, -.5f, .5f}, {.9f, .6f, .1f} }, //orange
        { {.5f, .5f, .5f}, {8.f, .1f, .8f} }, //magenta

            // tail face (green)
        { {-.5f, -.5f, -.5f}, { .9f, .9f, .9f } }, //white
        { {.5f, .5f, -.5f}, {.1f, .8f, .1f} }, //green
        { {-.5f, .5f, -.5f}, {.8f, .1f, .1f} }, //red
        { {-.5f, -.5f, -.5f}, { .9f, .9f, .9f } }, //white
        { {.5f, -.5f, -.5f}, {.8f, .8f, .1f} }, //yellow
        { {.5f, .5f, -.5f}, {.1f, .8f, .1f} }, //green

    };
    for (auto& v : vertices) {
        v.position += offset;
    }
    return std::make_unique<V_Model>(device, vertices);
}

void Vulkan::loadGameObjects() {
    std::shared_ptr<V_Model> v_model = createCubeModel(v_device, { .0f, .0f, .0f });

    auto cube = V_GameObject::createGameObject();
    cube.model = v_model;
    cube.transform.translation = { .0f, .0f, .5f };
    cube.transform.scale = { .5f, .5f, .5f };

    gameObjects.push_back(std::move(cube));
}

}