#include "cuda_render_system.hpp"

#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>

#include <vulkan/vulkan_win32.h>

#include "v_utils.hpp"

//namespace std {
//	template<>
//	struct hash<v::CudaRenderSystem::C_Vertex> {
//		size_t operator()(v::CudaRenderSystem::C_Vertex const& vertex) const {
//			size_t seed = 0;
//			v::hashCombine(seed, vertex.position, vertex.color, vertex.normal, vertex.uv);
//			return seed;
//		}
//	};
//}

namespace v {

HANDLE CudaRenderSystem::getVkSemaphoreHandle(
	VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
	VkSemaphore& semVkCuda) {
	HANDLE handle;

	VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
	vulkanSemaphoreGetWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
	vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = semVkCuda;
	vulkanSemaphoreGetWin32HandleInfoKHR.handleType =
		externalSemaphoreHandleType;

	vkGetSemaphoreWin32HandleKHR(v_device.device(), &vulkanSemaphoreGetWin32HandleInfoKHR,
		&handle);

	return handle;
}

void CudaRenderSystem::c_importVkSemaphore() {
	cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
	memset(&externalSemaphoreHandleDesc, 0,
		sizeof(externalSemaphoreHandleDesc));
	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, cudaUpdateVkSemaphore);
	externalSemaphoreHandleDesc.flags = 0;

	checkCudaErrors(cudaImportExternalSemaphore(&extCudaHandledSemaphore, &externalSemaphoreHandleDesc));

	externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
	externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, vkUpdateCudaSemaphore);
	externalSemaphoreHandleDesc.flags = 0;
	checkCudaErrors(cudaImportExternalSemaphore(&extVulkanHandledSemaphore, &externalSemaphoreHandleDesc));
	printf("CUDA Imported Vulkan semaphore\n");
}

HANDLE CudaRenderSystem::getVkMemoryHandle(VkDeviceMemory device_memory) {
	HANDLE handle = NULL;

	std::cout << device_memory << "\n";
	VkMemoryGetWin32HandleInfoKHR get_handle_info{ VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, VK_NULL_HANDLE, device_memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT };
	if (vkGetMemoryWin32HandleKHR(v_device.device(), &get_handle_info, &handle) != VK_SUCCESS) {
		std::cout << "getting memory handle failed\n";
	}
	std::cout << handle << "\n";
	if (!handle || handle == INVALID_HANDLE_VALUE) {
		std::cout << "bad handle\n";
	}

	return handle;
}

void CudaRenderSystem::c_importMemory(HANDLE memoryHandle, size_t extMemSize, VkDeviceMemory extMemory, void** bufferPtr, cudaExternalMemory_t &extBuffer) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	cudaExtMemHandleDesc.handle.win32.handle = memoryHandle;
	cudaExtMemHandleDesc.size = extMemSize;

	checkCudaErrors(cudaImportExternalMemory(&extBuffer,
		&cudaExtMemHandleDesc));

	cudaExternalMemoryBufferDesc bufferDesc = {};
	bufferDesc.offset = 0;
	bufferDesc.size = extMemSize; 
	bufferDesc.flags = 0;
	checkCudaErrors(cudaExternalMemoryGetMappedBuffer(bufferPtr, extBuffer, &bufferDesc));
}

void CudaRenderSystem::c_trace() {
	if (firstFrame) firstFrame = false;
	else c_waitVkSemaphore();

	vkQueueWaitIdle(v_device.graphicsQueue());
	/*void* indexBufferPtr;
	c_importMemory(getVkMemoryHandle(model->getIndexBufferMemory()), model->getIndexBufferSize(), &indexBufferPtr);*/
	c_signalVkSemaphore();

	// get buffers
	// for now just std::cout them
	//
	// signalVkSemaphore();
	// cudaFree(indexBufferPtr);
}

void CudaRenderSystem::c_signalVkSemaphore() {
	cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
	memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

	extSemaphoreSignalParams.params.fence.value = 0;
	extSemaphoreSignalParams.flags = 0;
	checkCudaErrors(cudaSignalExternalSemaphoresAsync(
		&extCudaHandledSemaphore, &extSemaphoreSignalParams, 1, streamToRun));
}

void CudaRenderSystem::c_waitVkSemaphore() {
	cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;

	memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));

	extSemaphoreWaitParams.params.fence.value = 0;
	extSemaphoreWaitParams.flags = 0;

	checkCudaErrors(cudaWaitExternalSemaphoresAsync(
		&extVulkanHandledSemaphore, &extSemaphoreWaitParams, 1, streamToRun));
	std::cout << "waited and good\n";
}



std::vector<VkVertexInputBindingDescription> CudaRenderSystem::C_Vertex::getBindingDescriptions() {
	std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
	bindingDescriptions[0].binding = 0;
	bindingDescriptions[0].stride = sizeof(C_Vertex);
	std::cout << ":DD\n";
	bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	return bindingDescriptions;
}


std::vector<VkVertexInputAttributeDescription> CudaRenderSystem::C_Vertex::getAttributeDescriptions() {
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};

	attributeDescriptions.push_back({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(C_Vertex, position) });
	attributeDescriptions.push_back({ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(C_Vertex, color) });
	attributeDescriptions.push_back({ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(C_Vertex, normal) });
	attributeDescriptions.push_back({ 3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(C_Vertex, uv) });
	return attributeDescriptions;
}

}