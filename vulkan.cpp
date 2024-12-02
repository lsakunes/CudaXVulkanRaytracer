#include "vulkan.hpp"
#include "keyboard_movement_controller.hpp"
#include <chrono>

namespace v {
__global__ void Vulkan::run() {
    SimpleRenderSystem simpleRenderSystem{ v_device, v_renderer.getSwapChainRenderPass() };
    //TODO: use cudarendersystem
    CudaRenderSystem cudaRenderSystem{ v_device };

    
    V_Camera camera{};
    camera.setViewTarget(glm::vec3(-1.f, -2.f, 2.f), glm::vec3(0.f, 0.f, 2.5f));

    auto viewerObject = V_GameObject::createGameObject();
    KeyboardMovementController cameraController{};

    std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();

    while (!v_window.shouldClose()) {
        glfwPollEvents();

        // give cuda the buffers here?
        // and get a handle for a semaphore

        std::chrono::steady_clock::time_point newTime = std::chrono::high_resolution_clock::now();
        float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
        currentTime = newTime;

        cameraController.moveInPlaneXZ(v_window.getGLFWwindow(), frameTime, viewerObject);
        camera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);

        float aspect = v_renderer.getAspectRatio();
        //camera.setOrthographicProjection(-aspect, aspect, -1, 1, -1, 1);
        camera.setPerspectiveProjection(glm::radians(50.f), aspect, 0.1f, 10.f);
        if (auto commandBuffer = v_renderer.beginFrame()) {
            v_renderer.beginSwapChainRenderPass(commandBuffer); 

            // on semaphore signal
            simpleRenderSystem.renderGameObjects(commandBuffer, gameObjects, camera); // TODO: swap out with something that accepts cuda's rendered image then displays it

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
    std::shared_ptr<V_Model> v_model = V_Model::createModelFromFile(v_device, "C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanInterop/models/suzanne.obj");

    V_GameObject gameObj = V_GameObject::createGameObject();
    gameObj.model = v_model;
    gameObj.transform.rotation = { 3.14f, 0.f, 0.f };
    gameObj.transform.translation = { .0f, .0f, 2.5f };
    gameObj.transform.scale = .5f;

    gameObjects.push_back(std::move(gameObj));
}
}