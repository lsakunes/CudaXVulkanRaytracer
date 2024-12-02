#include "cuda_render_system.hpp"

#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>

#include <vulkan/vulkan_win32.h>

namespace v {

HANDLE CudaRenderSystem::getVkMemoryHandle(VkDeviceMemory device_memory) {
	HANDLE handle;

	VkMemoryGetWin32HandleInfoKHR get_handle_info{ VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, VK_NULL_HANDLE, device_memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT };
	vkGetMemoryWin32HandleKHR(v_device.device(), &get_handle_info, &handle);
	 
	return handle;
}

void CudaRenderSystem::c_importMemory(HANDLE memoryHandle, size_t extMemSize, cudaExternalMemory_t& extMemory) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	cudaExtMemHandleDesc.handle.win32.handle = memoryHandle;
	cudaExtMemHandleDesc.size = extMemSize;

	checkCudaErrors(cudaImportExternalMemory(&extMemory,
		&cudaExtMemHandleDesc));

}

void CudaRenderSystem::c_trace(V_Model model) {
	c_importMemory(getVkMemoryHandle(model.getVertexBufferMemory()), model.getVertexBufferSize(), extVertexBuffer);
	c_importMemory(getVkMemoryHandle(model.getIndexBufferMemory()), model.getIndexBufferSize(), extIndexBuffer);
	// get buffers
	// for now just std::cout them
	//
	// signalVkSemaphore();
}

void CudaRenderSystem::c_signalVkSemaphore() {
	cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
	memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

	extSemaphoreSignalParams.params.fence.value = 0;
	extSemaphoreSignalParams.flags = 0;
	checkCudaErrors(cudaSignalExternalSemaphoresAsync(
		&extSemaphore, &extSemaphoreSignalParams, 1, streamToRun));
}


void CudaRenderSystem::c_allocateMemory(std::vector<V_GameObject>& gameObjects, const V_Camera& camera) {
	VkExportMemoryAllocateInfo export_info;
	export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT; // There's also nv and khr? i don't konw what those mean...

	VkMemoryAllocateInfo allocate_info;
	allocate_info.pNext = &export_info;

	VkDeviceMemory device_memory;
	vkAllocateMemory(v_device.device(), &allocate_info, nullptr, &device_memory);
}

}