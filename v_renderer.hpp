#ifndef RENDERERCPP
#define RENDERERCPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <array>

#include "v_window.hpp"
#include "v_device.hpp"
#include "v_swap_chain.hpp"
#include "v_model.hpp"

namespace v {

class V_Renderer {
public:

    V_Renderer(V_Window& window, V_Device& device);
    ~V_Renderer();

    V_Renderer(const V_Renderer&) = delete;
    V_Renderer& operator=(const V_Renderer&) = delete;

    VkRenderPass getSwapChainRenderPass() const { return v_swapchain->getRenderPass(); }
    bool isFrameInProgress() const { return isFrameStarted; }

    VkCommandBuffer getCurrentCommandBuffer() const {
        assert(isFrameStarted && "Cannot get command buffer when frame not in progress");
        return commandBuffers[currentImageIndex];
    }



    VkCommandBuffer beginFrame();
    void endFrame();
    void beginSwapChainRenderPass(VkCommandBuffer commandBuffer);
    void endSwapChainRenderPass(VkCommandBuffer commandBuffer);

private:
    void createCommandBuffers();
    void freeCommandBuffers();
    void recordCommandBuffer(int imageIndex);
    void recreateSwapChain();

    V_Window& v_window;
    V_Device& v_device;
    std::unique_ptr<V_SwapChain> v_swapchain; // performance hit using a pointer
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t currentImageIndex;
    bool isFrameStarted;
};


V_Renderer::V_Renderer(V_Window& window, V_Device& device) : v_window{ window }, v_device{ device } {
    createCommandBuffers();
}

V_Renderer::~V_Renderer() {
    freeCommandBuffers();
}

void V_Renderer::createCommandBuffers() {
    commandBuffers.resize(v_swapchain->imageCount());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = v_device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(v_device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Error allocating command buffers");
    }
}

void V_Renderer::recordCommandBuffer(int imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to being recording command buffer");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = v_swapchain->getRenderPass();
    renderPassInfo.framebuffer = v_swapchain->getFrameBuffer(imageIndex);

    renderPassInfo.renderArea.offset = { 0,0 };
    renderPassInfo.renderArea.extent = v_swapchain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0, 0, 0, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // TODO: contents inline might change to use secondary command buffers with cuda?? maybe??

    renderGameObjects(commandBuffers[imageIndex]);

    vkCmdEndRenderPass(commandBuffers[imageIndex]);
    if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer");
    }
}

void V_Renderer::recreateSwapChain() {
    auto extent = v_window.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        extent = v_window.getExtent();
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(v_device.device());
    v_swapchain = nullptr;
    v_swapchain = std::make_unique<V_SwapChain>(v_device, extent);
}


void V_Renderer::beginFrame() {
    uint32_t imageIndex;
    auto result = v_swapchain.acquireNextImage(&imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    recordCommandBuffer(imageIndex);
    result = v_swapchain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image");
    }
}

void V_Renderer::endFrame() {
    uint32_t imageIndex;
    auto result = v_swapchain.acquireNextImage(&imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    recordCommandBuffer(imageIndex);
    result = v_swapchain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image");
    }
}
}
#endif