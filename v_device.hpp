#ifndef VDEVICEHPP
#define VDEVICEHPP

#include <cstring>
#include <iostream>
#include <set>
#include <unordered_set>
#include "v_window.hpp"

// std lib headers
#include <string>
#include <vector>

namespace v {

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
  uint32_t graphicsFamily;
  uint32_t presentFamily;
  bool graphicsFamilyHasValue = false;
  bool presentFamilyHasValue = false;
  bool isComplete() { return graphicsFamilyHasValue && presentFamilyHasValue; }
};

class V_Device {
     public:
    #ifdef NDEBUG
      const bool enableValidationLayers = false;
    #else
      const bool enableValidationLayers = true;
    #endif

      V_Device(V_Window &window);
      ~V_Device();

      // Not copyable or movable
      V_Device(const V_Device &) = delete;
      void operator=(const V_Device &) = delete;
      V_Device(V_Device &&) = delete;
      V_Device &operator=(V_Device &&) = delete;

      VkCommandPool getCommandPool() { return commandPool; }
      VkDevice device() { return device_; }
      VkSurfaceKHR surface() { return surface_; }
      VkQueue graphicsQueue() { return graphicsQueue_; }
      VkQueue presentQueue() { return presentQueue_; }

      SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(physicalDevice); }
      uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
      QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(physicalDevice); }
      VkFormat findSupportedFormat(
          const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

      // Buffer Helper Functions
      void createBuffer(
          VkDeviceSize size,
          VkBufferUsageFlags usage,
          VkMemoryPropertyFlags properties,
          VkBuffer &buffer,
          VkDeviceMemory &bufferMemory);
  
      void createExtBuffer(
          VkDeviceSize size,
          VkBufferUsageFlags usage,
          VkMemoryPropertyFlags properties,
          VkBuffer &buffer,
          VkDeviceMemory &bufferMemory,
          VkExternalMemoryBufferCreateInfo &extBufferCreateInfo);
      VkCommandBuffer beginSingleTimeCommands();
      void endSingleTimeCommands(VkCommandBuffer commandBuffer);
      void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
      void copyBufferToImage(
          VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layerCount);

      void createImageWithInfo(
          const VkImageCreateInfo &imageInfo,
          VkMemoryPropertyFlags properties,
          VkImage &image,
          VkDeviceMemory &imageMemory);

      VkPhysicalDeviceProperties properties;

     private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    // helper functions
    bool isDeviceSuitable(VkPhysicalDevice device);
    std::vector<const char *> getRequiredExtensions();
    bool checkValidationLayerSupport();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void hasGflwRequiredInstanceExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    V_Window &window;
    VkCommandPool commandPool;

    VkDevice device_;
    VkSurfaceKHR surface_;
    VkQueue graphicsQueue_;
    VkQueue presentQueue_;

    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    // https://stackoverflow.com/questions/55424875/use-vulkan-vkimage-as-a-cuda-cuarray
    const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, //WE NEED EXTERNAL WIN32 EXTENSION THINGY
        "VK_KHR_external_semaphore_win32",
        "VK_KHR_external_memory_win32"
    };

            // My cards don't support these extensions; I'm not sure if I need them?
            // UPDATE: I def do need them for the forums I'm following, specifically memory file descriptors.
            // I'll try to look for how to do it alternatively 
            // VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME 
            // VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
            // 
            // WAIT 
            // FD IS FOR NON-WINDOWS PLATFORMS
            // I MIGHT BE GOOD
            // 
            // I need to us VK_KHR_external_memory_win32
            //
};

}  // namespace v

#endif