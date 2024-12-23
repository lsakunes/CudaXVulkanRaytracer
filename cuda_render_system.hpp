#ifndef CUDARENDERSYSTEM
#define CUDARENDERSYSTEM
#define VK_USE_PLATFORM_WIN32_KHR

#include "v_pipeline.hpp"
#include "v_gameobject.hpp"
#include "v_camera.hpp"
#include "cuda_runtime.h"
#include <windows.h>

#include <vulkan/vulkan_win32.h>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

namespace v {



class CudaRenderSystem {
public:
    struct C_Vertex { // TODO: restructuring memory to be multiple structs or smth might be faster
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal{};
        glm::vec2 uv{};

        static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
        static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();

        bool operator==(const C_Vertex& other) const {
            return position == other.position && color == other.color && normal == other.normal && uv == other.uv;
        }
    };

    void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
        if (result != cudaSuccess) {
            std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << " " << "' \n";
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }

    CudaRenderSystem(V_Device& device, VkSemaphore cudaToVkSemaphore, VkSemaphore vkToCudaSemaphore, std::vector<V_GameObject> *objects) : v_device(device), cudaUpdateVkSemaphore(cudaToVkSemaphore), vkUpdateCudaSemaphore(vkToCudaSemaphore), gameObjects(objects) {
        vkGetSemaphoreWin32HandleKHR =
            (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(v_device.device(), "vkGetSemaphoreWin32HandleKHR");
        if (vkGetSemaphoreWin32HandleKHR == NULL) {
            throw std::runtime_error(
                "Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not "
                "found.\n");
        }

        vkGetMemoryWin32HandleKHR =
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device.device(), "vkGetMemoryWin32HandleKHR");
        if (!vkGetMemoryWin32HandleKHR) {
            throw std::runtime_error("Failed to load vkGetMemoryWin32HandleKHR");
        }
        checkCudaErrors(cudaStreamCreate(&streamToRun));
        c_importVkSemaphore();

        std::shared_ptr<V_Model> model = (*gameObjects)[0].model;

        HANDLE extMemHandle = getVkMemoryHandle(model->getExtVertexBufferMemory());
        cudaExternalMemory_t extBuffer;
        c_importMemory(extMemHandle, model->getVertexBufferSize(), model->getExtVertexBufferMemory(), &vertexBufferPtr, extBuffer);



        cudaDestroyExternalMemory(extBuffer);
        CloseHandle(extMemHandle);
    }

    ~CudaRenderSystem() {
        checkCudaErrors(cudaDestroyExternalSemaphore(extVulkanHandledSemaphore));
        checkCudaErrors(cudaDestroyExternalSemaphore(extCudaHandledSemaphore));
        vkDestroySemaphore(v_device.device(), cudaUpdateVkSemaphore, nullptr);
        vkDestroySemaphore(v_device.device(), vkUpdateCudaSemaphore, nullptr);

    }

    CudaRenderSystem(const CudaRenderSystem&) = delete;
    CudaRenderSystem& operator=(const CudaRenderSystem&) = delete;


    void c_trace();

private:
    void c_signalVkSemaphore();
    HANDLE getVkMemoryHandle(VkDeviceMemory device_memory);
    void c_importMemory(HANDLE memoryHandle, size_t extMemSize, VkDeviceMemory extMemory, void** bufferPtr, cudaExternalMemory_t &extBuffer);
    HANDLE getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBits externalSemaphoreHandleType, VkSemaphore& semVkCuda);
    void c_importVkSemaphore();
    void c_waitVkSemaphore();
    void c_createImage(uint32_t width, uint32_t height, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, VkImage& image,
        VkDeviceMemory& imageMemory);


    cudaStream_t streamToRun; // TODO: figure out what the fuck this is
    // https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    V_Device& v_device;
    cudaExternalSemaphore_t extCudaHandledSemaphore;
    cudaExternalSemaphore_t extVulkanHandledSemaphore;
    VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;
    std::vector<V_GameObject> *gameObjects;

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR;




    void* vertexBufferPtr;
    bool firstFrame = true;
};

}

#endif