#include <stdio.h>
#include "vulkan.hpp"
using namespace v; 


//void importMemory() {
//    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = { };
//    memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
//    memDesc.handle.fd = getVulkanMemoryHandle(device, memory);
//    memDesc.size = extent.width * extent.height * 4;
//
//    CUDA_DRVAPI_CALL(cuImportExternalMemory(&externalMem, &memDesc));
//}

int main()
{
    v::Vulkan app{};
    app.run(); 

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


