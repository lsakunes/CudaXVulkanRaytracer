#include "cuda_render_system.hpp"

#include "windowsSecurity.hpp"
#include <cstring>
#include <cassert>
#include "kernel.cuh"

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

HANDLE CudaRenderSystem::getVkMemoryHandle(VkDeviceMemory device_memory) {
	HANDLE handle = NULL;
	std::cout << device_memory << "\n";
	VkMemoryGetWin32HandleInfoKHR get_handle_info{ VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, VK_NULL_HANDLE, device_memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT };
	vkGetMemoryWin32HandleKHR(v_device.device(), &get_handle_info, &handle);

	if (vkGetMemoryWin32HandleKHR(v_device.device(), &get_handle_info, &handle) != VK_SUCCESS) {
		std::cout << "getting memory handle failed\n";
	}
	std::cout << handle << "\n";
	if (!handle || handle == INVALID_HANDLE_VALUE) {
		std::cout << "bad handle\n";
	}
	return handle;
}

uint32_t CudaRenderSystem::c_findMemoryType(uint32_t typeFilter,
	VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(v_device.physicalDevice(), &memProperties);

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

	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArray, cudaMipmappedImageArray, 0));
	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArrayTemp, cudaMipmappedImageArrayTemp, 0));
	checkCudaErrors(cudaGetMipmappedArrayLevel(
		&cudaMipLevelArrayOrig, cudaMipmappedImageArrayOrig, 0));
	checkCudaErrors(cudaMemcpy2DArrayToArray(
		cudaMipLevelArrayOrig, 0, 0, cudaMipLevelArray, 0, 0,
		width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice)); // TODO: investigate whether it's supposed to be uchar or smth else

	cudaResourceDesc resourceDesc;
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

	// cudaTextureDesc texDescr;
	//memset(&texDescr, 0, sizeof(cudaTextureDesc));

	//texDescr.normalizedCoords = true;
	//texDescr.filterMode = cudaFilterModeLinear;
	//texDescr.mipmapFilterMode = cudaFilterModeLinear;

	//texDescr.addressMode[0] = cudaAddressModeWrap;
	//texDescr.addressMode[1] = cudaAddressModeWrap;

	//texDescr.maxMipmapLevelClamp = 0;

	//texDescr.readMode = cudaReadModeNormalizedFloat;

	// checkCudaErrors(cudaCreateTextureObject(&textureObjMipMapInput, &resDescr,
	//	&texDescr, NULL)); // TODO: I think i have to use this somewhere??? 

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

void CudaRenderSystem::c_createSampler() {
	VkSamplerCreateInfo samplerCreateInfo{};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	if (vkCreateSampler(v_device.device(), &samplerCreateInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create sampler!");
	}
}

void CudaRenderSystem::c_createImageView() {
	VkImageViewCreateInfo imageViewCreateInfo{};
	imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCreateInfo.image = image;
	imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	imageViewCreateInfo.subresourceRange.levelCount = 1;
	imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	imageViewCreateInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(v_device.device(), &imageViewCreateInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image view!");
	}
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

void CudaRenderSystem::c_trace(VkCommandBuffer commandBuffer, int frameIndex) {
	if (firstFrame) firstFrame = false;
	else c_waitVkSemaphore();
	launchPlainUV(height, width, streamToRun, d_surfaceObject);
	c_signalVkSemaphore(); // TODO: later signal semaphore async? 
	// c_updateDescriptorSets();
	v_pipeline->bindGraphics(commandBuffer);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);
	vkCmdDraw(commandBuffer, 6, 1, 0, 0);
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

void CudaRenderSystem::c_createPipelineBarrier(VkCommandBuffer commandBuffer) {
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask =  VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
}

void CudaRenderSystem::c_createPipelineLayout() {

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	if (vkCreatePipelineLayout(v_device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
		VK_SUCCESS) {
		throw std::runtime_error("error at createPipelineLayoutInfo");
	}
}

void CudaRenderSystem::c_createPipeline(VkRenderPass renderPass) {
	assert(pipelineLayout != nullptr && "Cannot create pipeline before pipline layout");

	PipelineConfigInfo pipelineConfig = PipelineConfigInfo{};
	V_Pipeline::defaultPipelineConfigInfo(pipelineConfig);

	pipelineConfig.renderPass = renderPass;
	pipelineConfig.pipelineLayout = pipelineLayout;

	v_pipeline = std::make_unique<V_Pipeline>(
		v_device,
		"C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanRaytracer/shaders/vert.spv",
		"C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanRaytracer/shaders/frag.spv",
		pipelineConfig);
}

void CudaRenderSystem::c_createDescriptorSets() {
	std::vector<VkDescriptorSetLayout> layouts(numSwapChainImages,
		descriptorSetLayout);
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = numSwapChainImages;
	allocInfo.pSetLayouts = layouts.data();

	descriptorSets.resize(numSwapChainImages);

	if (vkAllocateDescriptorSets(v_device.device(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate descriptor set!");
	}

}

void CudaRenderSystem::c_updateDescriptorSets() {
	for (size_t i = 0; i < numSwapChainImages; i++) {
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = imageView;
		imageInfo.sampler = sampler;

		VkWriteDescriptorSet writeDescriptorSet{};
		writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet.dstSet = descriptorSets[i];
		writeDescriptorSet.dstBinding = 0;
		writeDescriptorSet.dstArrayElement = 0;
		writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writeDescriptorSet.descriptorCount = 1;
		writeDescriptorSet.pImageInfo = &imageInfo;

		vkUpdateDescriptorSets(v_device.device(), 1, &writeDescriptorSet, 0, nullptr);
	}
}

void CudaRenderSystem::c_createDescriptorPool() {
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	poolSize.descriptorCount = numSwapChainImages;

	VkDescriptorPoolCreateInfo poolCreateInfo{};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.poolSizeCount = 1;
	poolCreateInfo.pPoolSizes = &poolSize;
	poolCreateInfo.maxSets = numSwapChainImages;
	poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
	if (vkCreateDescriptorPool(v_device.device(), &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor pool!");
	}
}

void CudaRenderSystem::c_createDescriptorSetLayout() {
	VkDescriptorSetLayoutBindingFlagsCreateInfoEXT bindingFlagsInfo{};
	bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
	bindingFlagsInfo.bindingCount = 1;
	VkDescriptorBindingFlagsEXT bindingFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
	bindingFlagsInfo.pBindingFlags = &bindingFlags;

	VkDescriptorSetLayoutBinding imageBinding{};
	imageBinding.binding = 0;
	imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	imageBinding.descriptorCount = 1;
	imageBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; 
	imageBinding.pImmutableSamplers = nullptr;

	// Descriptor set layout creation
	VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = 1;
	layoutCreateInfo.pBindings = &imageBinding;
	layoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
	layoutCreateInfo.pNext = &bindingFlagsInfo;

	if (vkCreateDescriptorSetLayout(v_device.device(), &layoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor set layout!");
	}

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
