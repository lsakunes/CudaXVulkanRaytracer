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
    void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
        if (result != cudaSuccess) {
            std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << " " << "' \n";
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }

    CudaRenderSystem(V_Device& device) : v_device(device) {
        vkGetMemoryWin32HandleKHR =
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device.device(), "vkGetMemoryWin32HandleKHR");
        if (!vkGetMemoryWin32HandleKHR) {
            throw std::runtime_error("Failed to load vkGetMemoryWin32HandleKHR");
        }
        //checkCudaErrors(cudaStreamCreate(&streamToRun));

        //cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
        //memset(&externalSemaphoreHandleDesc, 0,
        //    sizeof(externalSemaphoreHandleDesc));

        //externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        //externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle();

        //checkCudaErrors(cudaImportExternalSemaphore(&extSemaphore,
        //    &externalSemaphoreHandleDesc));
    }
    // ~CudaRenderSystem(); I don't think i need this

    CudaRenderSystem(const CudaRenderSystem&) = delete;
    CudaRenderSystem& operator=(const CudaRenderSystem&) = delete;


    void c_trace(V_Model model);

private:
    void c_signalVkSemaphore();
    HANDLE getVkMemoryHandle(VkDeviceMemory device_memory);
    void c_importMemory(HANDLE memoryHandle, size_t extMemSize, cudaExternalMemory_t& extMemory);
    HANDLE getVkSemaphoreHandle();

    void c_allocateMemory(std::vector<V_GameObject>& gameObjects, const V_Camera& camera);

    cudaStream_t streamToRun; // TODO: figure out what the fuck this is
    // https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
    V_Device& v_device;
    cudaExternalSemaphore_t extSemaphore;
    cudaExternalMemory_t extVertexBuffer;
    cudaExternalMemory_t extIndexBuffer;

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
};

}

#endif