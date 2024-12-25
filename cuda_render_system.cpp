#include "cuda_render_system.hpp"


#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#include <surface_functions.h>

#include <vulkan/vulkan_win32.h>
#include "windowsSecurity.hpp"
#include <cstring>

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

__device__ void plainUV(cudaSurfaceObject_t surface, int nWidth, int nHeight){
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);
	if (x + 1 >= nWidth || y + 1 >= nHeight) {
		return;
	}
	float value = static_cast<float>(x + y); // Example value
	surf2Dwrite(value, surface, x * sizeof(float), y);
}

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

uint32_t CudaRenderSystem::c_findMemoryType(uint32_t typeFilter,
	VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(v_device.physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) &&
			(memProperties.memoryTypes[i].propertyFlags & properties) ==
			properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void CudaRenderSystem::c_createImage() {
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = static_cast<uint32_t>(width);
	imageInfo.extent.height = static_cast<uint32_t>(height);
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
	vkExternalMemImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	vkExternalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	imageInfo.pNext = &vkExternalMemImageCreateInfo;

	if (vkCreateImage(v_device.device(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(v_device.device(), image, &memRequirements);


	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	imageMemSize = memRequirements.size;

	VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };

	WindowsSecurityAttributes win_security_attributes;
	VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR };

	vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &win_security_attributes;
	vulkanExportMemoryWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;

	vulkanExportMemoryAllocateInfoKHR.pNext = &vulkanExportMemoryWin32HandleInfoKHR;
	vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;

	VkMemoryPropertyFlags flags = 0;
	flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	allocInfo.memoryTypeIndex = c_findMemoryType(memRequirements.memoryTypeBits, flags);

	if (vkAllocateMemory(v_device.device(), &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}
	vkBindImageMemory(v_device.device(), image, imageMemory, 0);

	VkMemoryGetWin32HandleInfoKHR desc{};
	desc.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	desc.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	desc.memory = imageMemory;

	vkGetMemoryWin32HandleKHR(v_device.device(), &desc, &imageHandle);
}

HANDLE CudaRenderSystem::getVkImageMemHandle(
	VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) {
	HANDLE handle;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
	vkMemoryGetWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory = imageMemory;
	vkMemoryGetWin32HandleInfoKHR.handleType =
		(VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

	vkGetMemoryWin32HandleKHR(v_device.device(), &vkMemoryGetWin32HandleInfoKHR, &handle);
	return handle;
}

void CudaRenderSystem::c_importImage() {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
	cudaExtMemHandleDesc.size = imageMemSize;

	checkCudaErrors(cudaImportExternalMemory(&cudaExtMemImageBuffer,
		&cudaExtMemHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;

	memset(&externalMemoryMipmappedArrayDesc, 0,
		sizeof(externalMemoryMipmappedArrayDesc));

	cudaExtent extent = make_cudaExtent(width, height, 0);
	cudaChannelFormatDesc formatDesc;
	formatDesc.x = 8;
	formatDesc.y = 8;
	formatDesc.z = 8;
	formatDesc.w = 8;
	formatDesc.f = cudaChannelFormatKindUnsigned;

	externalMemoryMipmappedArrayDesc.offset = 0;
	externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
	externalMemoryMipmappedArrayDesc.extent = extent;
	externalMemoryMipmappedArrayDesc.flags = 0;
	externalMemoryMipmappedArrayDesc.numLevels = 1;

	checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
		&cudaMipmappedImageArray, cudaExtMemImageBuffer,
		&externalMemoryMipmappedArrayDesc));

	checkCudaErrors(cudaMallocMipmappedArray(&cudaMipmappedImageArrayTemp,
		&formatDesc, extent, 1));
	checkCudaErrors(cudaMallocMipmappedArray(&cudaMipmappedImageArrayOrig,
		&formatDesc, extent, 1));
	cudaArray_t cudaMipLevelArray, cudaMipLevelArrayTemp,
		cudaMipLevelArrayOrig;
	cudaResourceDesc resourceDesc;

	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArray, cudaMipmappedImageArray, 1));
	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArrayTemp, cudaMipmappedImageArrayTemp, 1));
	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArrayOrig, cudaMipmappedImageArrayOrig, 1));
	checkCudaErrors(cudaMemcpy2DArrayToArray(
		cudaMipLevelArrayOrig, 0, 0, cudaMipLevelArray, 0, 0,
		(width >> 1) * sizeof(uchar4), (height >> 1), cudaMemcpyDeviceToDevice));

	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = cudaMipLevelArray;

	checkCudaErrors(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));

	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = cudaMipLevelArrayTemp;

	checkCudaErrors(
		cudaCreateSurfaceObject(&surfaceObjectTemp, &resourceDesc));

	cudaResourceDesc resDescr;
	memset(&resDescr, 0, sizeof(cudaResourceDesc));

	resDescr.resType = cudaResourceTypeMipmappedArray;
	resDescr.res.mipmap.mipmap = cudaMipmappedImageArrayOrig;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.mipmapFilterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.addressMode[1] = cudaAddressModeWrap;

	texDescr.maxMipmapLevelClamp = 0;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&textureObjMipMapInput, &resDescr,
		&texDescr, NULL));

	checkCudaErrors(cudaMalloc((void**)&d_surfaceObject,
		sizeof(cudaSurfaceObject_t)));
	checkCudaErrors(cudaMalloc((void**)&d_surfaceObjectTemp,
		sizeof(cudaSurfaceObject_t)));

	checkCudaErrors(cudaMemcpy(d_surfaceObject, &surfaceObject,
		sizeof(cudaSurfaceObject_t),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_surfaceObjectTemp, &surfaceObjectTemp,
		sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice));

	printf("CUDA Kernel Vulkan image buffer\n");
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