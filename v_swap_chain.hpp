#pragma once

#include "v_device.hpp"

// vulkan headers
#include <vulkan/vulkan.h>

// std lib headers
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace v {

class V_SwapChain {
 public:
  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  V_SwapChain(V_Device &deviceRef, VkExtent2D windowExtent);
  V_SwapChain(V_Device& deviceRef, VkExtent2D windowExtent, std::shared_ptr<V_SwapChain> previous);
  ~V_SwapChain();

  V_SwapChain(const V_SwapChain &) = delete;
  void operator=(const V_SwapChain &) = delete;

  VkFramebuffer getFrameBuffer(int index) { return swapChainFramebuffers[index]; }
  VkRenderPass getRenderPass() { return renderPass; }
  VkImageView getImageView(int index) { return swapChainImageViews[index]; }
  size_t imageCount() { return swapChainImages.size(); }
  VkFormat getSwapChainImageFormat() { return swapChainImageFormat; }
  VkExtent2D getSwapChainExtent() { return swapChainExtent; }
  VkSwapchainKHR getSwapChainKHR() { return swapChain; }
  uint32_t width() { return swapChainExtent.width; }
  uint32_t height() { return swapChainExtent.height; }
  void setExtSemaphores(VkSemaphore cudaHandledSemaphore, VkSemaphore vkHandledSemaphore) { cudaUpdateVkSemaphore = cudaHandledSemaphore; vkUpdateCudaSemaphore = vkHandledSemaphore; }
  VkSemaphore getCudaHandledSemaphore() { return cudaUpdateVkSemaphore; }
  VkSemaphore getVkHandledSemaphore() { return vkUpdateCudaSemaphore; }

  float extentAspectRatio() {
    return static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
  }
  VkFormat findDepthFormat();

  VkResult acquireNextImage(uint32_t *imageIndex);
  VkResult submitCommandBuffers(const VkCommandBuffer *buffers, uint32_t *imageIndex);

  bool compareSwapFormats(const V_SwapChain& swapChain) const {
      return swapChain.swapChainDepthFormat == swapChainDepthFormat && swapChain.swapChainImageFormat == swapChainImageFormat;
  }

 private:
  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createFramebuffers();
  void createSyncObjects();
  void init();

  // Helper functions
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats);
  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

  VkFormat swapChainImageFormat;
  VkFormat swapChainDepthFormat;
  VkExtent2D swapChainExtent;

  std::vector<VkFramebuffer> swapChainFramebuffers;
  VkRenderPass renderPass;

  std::vector<VkImage> depthImages;
  std::vector<VkDeviceMemory> depthImageMemorys;
  std::vector<VkImageView> depthImageViews;
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;

  V_Device &device;
  VkExtent2D windowExtent;

  VkSwapchainKHR swapChain;
  std::shared_ptr<V_SwapChain> oldSwapChain;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  std::vector<VkFence> imagesInFlight;
  VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;
  size_t currentFrame = 0;
};
}  // namespace lve
