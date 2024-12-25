#ifndef CUDARENDERSYSTEM
#define CUDARENDERSYSTEM
#define VK_USE_PLATFORM_WIN32_KHR

#include "v_pipeline.hpp"
#include "v_gameobject.hpp"
#include "v_camera.hpp"
#include <cuda_runtime.h>
#include <windows.h>
#include "device_launch_parameters.h"
#include <iostream>

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

    CudaRenderSystem(
        V_Device& device, VkSemaphore cudaToVkSemaphore, VkSemaphore vkToCudaSemaphore,
        std::vector<V_GameObject> *objects, uint32_t h, uint32_t w) : height(h), width(w),
        v_device(device), cudaUpdateVkSemaphore(cudaToVkSemaphore), 
        vkUpdateCudaSemaphore(vkToCudaSemaphore), gameObjects(objects) {

        vkGetSemaphoreWin32HandleKHR =
            (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(v_device.device(), "vkGetSemaphoreWin32HandleKHR");
        if (vkGetSemaphoreWin32HandleKHR == NULL) {
            throw std::runtime_error(
                "Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not "
                "found.\n");
        }

        vkGetMemoryWin32HandleKHR =
            (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(
                v_device.instance(), "vkGetMemoryWin32HandleKHR");
        if (vkGetMemoryWin32HandleKHR == NULL) {
            throw std::runtime_error(
                "Vulkan: Proc address for \"vkGetMemoryWin32HandleKHR\" not "
                "found.\n");
        }

        //vkGetMemoryWin32HandleKHR =
        //    (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device.device(), "vkGetMemoryWin32HandleKHR");
        //if (!vkGetMemoryWin32HandleKHR) {
        //    throw std::runtime_error("Failed to load vkGetMemoryWin32HandleKHR");
        //}
        checkCudaErrors(cudaStreamCreate(&streamToRun));
        c_importVkSemaphore();

        std::shared_ptr<V_Model> model = (*gameObjects)[0].model;

        HANDLE extMemHandle = getVkMemoryHandle(model->getExtVertexBufferMemory());
        cudaExternalMemory_t extBuffer;
        c_importMemory(extMemHandle, model->getVertexBufferSize(), model->getExtVertexBufferMemory(), &vertexBufferPtr, extBuffer);



        cudaDestroyExternalMemory(extBuffer);
        CloseHandle(extMemHandle);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaMallocArray(&colorArray, &channelDesc, width, height);
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = colorArray;

        cudaCreateSurfaceObject(&surfaceObj, &resDesc);
    }

    ~CudaRenderSystem() {
        checkCudaErrors(cudaDestroyExternalSemaphore(extVulkanHandledSemaphore));
        checkCudaErrors(cudaDestroyExternalSemaphore(extCudaHandledSemaphore));
        //checkCudaErrors(cudaDestroyExternalMemory(cudaExtMemImageBuffer)); //TODO: remember to uncomment this when we start using it
        checkCudaErrors(cudaDestroySurfaceObject(surfaceObject));
        checkCudaErrors(cudaDestroySurfaceObject(surfaceObjectTemp));
        checkCudaErrors(cudaFree(d_surfaceObject));
        checkCudaErrors(cudaFree(d_surfaceObjectTemp));
        checkCudaErrors(cudaFreeMipmappedArray(cudaMipmappedImageArrayTemp));
        checkCudaErrors(cudaFreeMipmappedArray(cudaMipmappedImageArrayOrig));
        checkCudaErrors(cudaFreeMipmappedArray(cudaMipmappedImageArray));
        vkDestroySemaphore(v_device.device(), cudaUpdateVkSemaphore, nullptr);
        vkDestroySemaphore(v_device.device(), vkUpdateCudaSemaphore, nullptr);
        cudaDestroySurfaceObject(surfaceObj);
        cudaFreeArray(colorArray);
        checkCudaErrors(cudaDestroyTextureObject(textureObjMipMapInput));

        vkDestroyImage(v_device.device(), image, nullptr);
        vkFreeMemory(v_device.device(), imageMemory, nullptr);

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
    uint32_t c_findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void c_createImage();
    void c_importImage();
    HANDLE getVkImageMemHandle(
        VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);

    cudaStream_t streamToRun; // TODO: figure out what the fuck this is
    // https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    V_Device& v_device;
    cudaExternalSemaphore_t extCudaHandledSemaphore;
    cudaExternalSemaphore_t extVulkanHandledSemaphore;
    VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;
    std::vector<V_GameObject> *gameObjects;

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR;


    cudaSurfaceObject_t surfaceObject, surfaceObjectTemp;
    cudaSurfaceObject_t* d_surfaceObject, * d_surfaceObjectTemp;

    cudaExternalMemory_t cudaExtMemImageBuffer;
    cudaMipmappedArray_t cudaMipmappedImageArray, cudaMipmappedImageArrayTemp,
        cudaMipmappedImageArrayOrig;
    cudaTextureObject_t textureObjMipMapInput;


    VkImage image;
    VkDeviceMemory imageMemory;
    HANDLE imageHandle;
    uint32_t imageMemSize;

    cudaArray_t colorArray;
    cudaSurfaceObject_t surfaceObj;


    uint32_t height, width;


    void* vertexBufferPtr;
    bool firstFrame = true;
};

}

#endif