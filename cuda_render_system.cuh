#ifndef CUDARENDERSYSTEM
#define CUDARENDERSYSTEM
#define VK_USE_PLATFORM_WIN32_KHR

#include "v_pipeline.hpp"
#include "v_gameobject.hpp"
#include "v_camera.hpp"
#include <cuda_runtime.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>
//#include "kernel.cuh"
#include "cudaMain.cu"

#define STB_IMAGE_IMPLEMENTATION 
#include "stb_image.h" 

#include <vulkan/vulkan_win32.h>

#ifdef __INTELLISENSE__
#define LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) kernel
#else
#define LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif

namespace v {

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

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

    CudaRenderSystem(
        V_Device& device, uint32_t nSwapChainImages, VkSemaphore cudaToVkSemaphore, VkSemaphore vkToCudaSemaphore,
        std::vector<V_GameObject>* objects, uint32_t h, uint32_t w, VkRenderPass renderPass);

    ~CudaRenderSystem();

    void c_recreateEverything() {

    }

    CudaRenderSystem(const CudaRenderSystem&) = delete;
    CudaRenderSystem& operator=(const CudaRenderSystem&) = delete;

    void c_createPipelineBarrier(VkCommandBuffer commandBuffer);

    void c_trace(VkCommandBuffer commandBuffer, int frameIndex);

    void c_moveCamera(glm::vec3 translation, glm::vec3 rotation);
private:
    void c_signalVkSemaphore();
    HANDLE getVkMemoryHandle(VkDeviceMemory device_memory);
    void c_importMemory(HANDLE memoryHandle, size_t extMemSize, VkDeviceMemory extMemory, void** bufferPtr, cudaExternalMemory_t& extBuffer);
    HANDLE getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBits externalSemaphoreHandleType, VkSemaphore& semVkCuda);
    void c_importVkSemaphore();
    void c_waitVkSemaphore();
    uint32_t c_findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void c_createImage();
    void c_importImage();
    void c_createFunctions() {
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
    }
    void c_createModel() {
        std::shared_ptr<V_Model> model = (*gameObjects)[0].model;
        HANDLE extMemHandle = getVkMemoryHandle(model->getExtVertexBufferMemory());
        cudaExternalMemory_t extBuffer;
        c_importMemory(extMemHandle, model->getVertexBufferSize(), model->getExtVertexBufferMemory(), &vertexBufferPtr, extBuffer);
        cudaDestroyExternalMemory(extBuffer);
        CloseHandle(extMemHandle);
    }
    void c_createSurfaceAndColorArray() {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaMallocArray(&colorArray, &channelDesc, width, height);
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = colorArray;

        cudaCreateSurfaceObject(&surfaceObj, &resDesc);
    }
    void c_createPipelineLayout();
    void c_createPipeline(VkRenderPass renderPass);
    void c_createDescriptorSetLayout();
    void c_createDescriptorSets();
    void c_createDescriptorPool();
    void c_updateDescriptorSets();
    void c_createSampler();
    void c_createImageView();
    HANDLE getVkImageMemHandle(
        VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);

    cudaStream_t streamToRun; // TODO: figure out what the fuck this is
    // https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    V_Device& v_device;
    cudaExternalSemaphore_t extCudaHandledSemaphore;
    cudaExternalSemaphore_t extVulkanHandledSemaphore;
    VkSemaphore cudaUpdateVkSemaphore, vkUpdateCudaSemaphore;
    std::vector<V_GameObject>* gameObjects;
    uint32_t numSwapChainImages;

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR;


    cudaSurfaceObject_t surfaceObject, surfaceObjectTemp;
    cudaSurfaceObject_t* d_surfaceObject, * d_surfaceObjectTemp;

    cudaExternalMemory_t cudaExtMemImageBuffer;
    cudaMipmappedArray_t cudaMipmappedImageArray, cudaMipmappedImageArrayTemp,
        cudaMipmappedImageArrayOrig;
    // cudaTextureObject_t textureObjMipMapInput;

    VkImage image;
    VkDeviceMemory imageMemory;
    HANDLE imageHandle;
    uint32_t imageMemSize;
    VkImageView imageView;

    cudaArray_t colorArray;
    cudaSurfaceObject_t surfaceObj;
    VkSampler sampler;

    VkPipelineLayout pipelineLayout;
    std::unique_ptr<V_Pipeline> v_pipeline;

    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;


    curandState* d_rand_state;

    camera** d_cam;
    hitable** d_list;
    hitable** d_world;
    unsigned char* d_tex_data;
    uint32_t samples;

    dim3 blocks;
    dim3 threads;


    vec3* ranvec;
    int* perm_x;
    int* perm_z;
    int* perm_y;


    uint32_t height, width;


    void* vertexBufferPtr;
    bool firstFrame = true;

    unsigned char* tex_data;
};
}

#endif