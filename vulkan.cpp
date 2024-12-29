#include "vulkan.hpp"
#include "keyboard_movement_controller.hpp"
#include <chrono>

namespace v {
__global__ void Vulkan::run() {
    CudaRenderSystem cudaRenderSystem{ v_device, V_SwapChain::MAX_FRAMES_IN_FLIGHT, cudaUpdateVkSemaphore, vkUpdateCudaSemaphore, &gameObjects, v_window.getExtent().height, v_window.getExtent().width, v_renderer.getSwapChainRenderPass()};


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
        cudaRenderSystem.c_moveCamera(viewerObject.transform.translation, viewerObject.transform.rotation);

        float aspect = v_renderer.getAspectRatio();
        if (auto commandBuffer = v_renderer.beginFrame()) {
            cudaRenderSystem.c_createPipelineBarrier(commandBuffer);
            v_renderer.beginSwapChainRenderPass(commandBuffer);
            cudaRenderSystem.c_trace(v_renderer.getCurrentCommandBuffer(),
                v_renderer.getFrameIndex());
            v_renderer.endSwapChainRenderPass(commandBuffer);
            v_renderer.endFrame();
        }
    }
    
     vkDeviceWaitIdle(v_device.device());
}

void Vulkan::createSyncObjectsExt() {
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    WindowsSecurityAttributes winSecurityAttributes;

    VkExportSemaphoreWin32HandleInfoKHR
        vulkanExportSemaphoreWin32HandleInfoKHR = {};
    vulkanExportSemaphoreWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    vulkanExportSemaphoreWin32HandleInfoKHR.pNext = NULL;
    vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes =
        &winSecurityAttributes;
    vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkanExportSemaphoreWin32HandleInfoKHR.name = (LPCWSTR)NULL;
    VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
    vulkanExportSemaphoreCreateInfo.sType =
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    vulkanExportSemaphoreCreateInfo.pNext = &vulkanExportSemaphoreWin32HandleInfoKHR;
    vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

    if (vkCreateSemaphore(v_device.device(), &semaphoreInfo, nullptr,
        &cudaUpdateVkSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(v_device.device(), &semaphoreInfo, nullptr,
            &vkUpdateCudaSemaphore) != VK_SUCCESS) {
        throw std::runtime_error(
            "failed to create synchronization objects for a CUDA-Vulkan!");
    }
}


Vulkan::Vulkan() {
    createSyncObjectsExt();
    loadGameObjects();
    v_renderer.setSwapchainExt(cudaUpdateVkSemaphore, vkUpdateCudaSemaphore);
}

Vulkan::~Vulkan() {}

void Vulkan::loadGameObjects() {
    std::shared_ptr<V_Model> v_model = V_Model::createModelFromFile(v_device, "C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanRaytracer/models/suzanne.obj");

    V_GameObject gameObj = V_GameObject::createGameObject();
    gameObj.model = v_model;
    gameObj.transform.rotation = { 3.14f, 0.f, 0.f };
    gameObj.transform.translation = { .0f, .0f, 2.5f };
    gameObj.transform.scale = .5f;

    gameObjects.push_back(std::move(gameObj));
}
}