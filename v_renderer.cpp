#include "v_renderer.hpp"

namespace v {

V_Renderer::V_Renderer(V_Window& window, V_Device& device) : v_window{ window }, v_device{ device } {
    recreateSwapChain();
    createCommandBuffers();
}

V_Renderer::~V_Renderer() {
    freeCommandBuffers();
}

void V_Renderer::freeCommandBuffers() {
    vkFreeCommandBuffers(
        v_device.device(),
        v_device.getCommandPool(),
        static_cast<uint32_t>(commandBuffers.size()),
        commandBuffers.data());
    commandBuffers.clear();
}

void V_Renderer::createCommandBuffers() {
    commandBuffers.resize(V_SwapChain::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = v_device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(v_device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Error allocating command buffers");
    }
}

void V_Renderer::recreateSwapChain() {
    auto extent = v_window.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        extent = v_window.getExtent();
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(v_device.device());

    if (v_swapchain == nullptr) {
        v_swapchain = std::make_unique < V_SwapChain>(v_device, extent);
    }
    else {
        std::shared_ptr<V_SwapChain> oldSwapChain = std::move(v_swapchain);
        v_swapchain = std::make_unique<V_SwapChain>(v_device, extent, oldSwapChain);
        if (!oldSwapChain->compareSwapFormats(*v_swapchain.get())) {
            throw std::runtime_error("Swap chain image (or depth) format has changed");
        }
    }
}


VkCommandBuffer V_Renderer::beginFrame() {
    assert(!isFrameStarted && "Can't call beginFrame while already in progress");

    auto result = v_swapchain->acquireNextImage(&currentImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return nullptr;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    isFrameStarted = true;

    auto commandBuffer = getCurrentCommandBuffer(); 
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to being recording command buffer");
    }
    return commandBuffer;
}

void V_Renderer::endFrame() {
    assert(isFrameStarted && "Can't call endFrame while frame is not in progress");
    auto commandBuffer = getCurrentCommandBuffer();
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to end command buffer");
    }

    auto result = v_swapchain->submitCommandBuffers(&commandBuffer, &currentImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || v_window.wasWindowResized()) {
        v_window.resetWindowResizedFlag();
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image");
    }

    isFrameStarted = false;
    currentFrameIndex = (currentFrameIndex + 1) % V_SwapChain::MAX_FRAMES_IN_FLIGHT;
}


void V_Renderer::beginSwapChainRenderPass(VkCommandBuffer commandBuffer) {
    assert(isFrameStarted && "Can't call beginSwapChainRenderPass if frame is not in progress");
    assert(commandBuffer == getCurrentCommandBuffer() && "Can't begin render pass on command buffer from a different frame");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = v_swapchain->getRenderPass();
    renderPassInfo.framebuffer = v_swapchain->getFrameBuffer(currentImageIndex);

    renderPassInfo.renderArea.offset = { 0,0 };
    renderPassInfo.renderArea.extent = v_swapchain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0, 0, 0, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); 

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(v_swapchain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(v_swapchain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{ {0,0}, v_swapchain->getSwapChainExtent() };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}
void V_Renderer::endSwapChainRenderPass(VkCommandBuffer commandBuffer) {
    assert(isFrameStarted && "Can't call endSwapChainRenderPass if frame is not in progress");
    assert(commandBuffer == getCurrentCommandBuffer() && "Can't end render pass on command buffer from a different frame");

    vkCmdEndRenderPass(commandBuffer);
}
}