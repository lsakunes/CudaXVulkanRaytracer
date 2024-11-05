#ifndef RENDERERCPP
#define RENDERERCPP

#include <cassert>
#include "v_swap_chain.hpp"

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
        return commandBuffers[currentFrameIndex];
    }



    VkCommandBuffer beginFrame();
    void endFrame();
    void beginSwapChainRenderPass(VkCommandBuffer commandBuffer);
    void endSwapChainRenderPass(VkCommandBuffer commandBuffer);

    int getFrameIndex() const {
        assert(isFrameStarted && "Can't call getFrameIndex while frame is not in progress");
        return currentFrameIndex;
    }

private:
    void createCommandBuffers();
    void freeCommandBuffers();
    void recreateSwapChain();

    V_Window& v_window;
    V_Device& v_device;
    std::unique_ptr<V_SwapChain> v_swapchain; // TODO: performance hit using a pointer
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t currentImageIndex;
    int currentFrameIndex = 0;
    bool isFrameStarted = false;
};
}
#endif