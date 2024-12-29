#include "cuda_render_system.cuh"

#include "windowsSecurity.hpp"
#include <cstring>
#include <cassert>
#include "kernel.cuh"
#include "v_gameobject.hpp"
#include "v_utils.hpp"


#define EPSILON 0.01f

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

CudaRenderSystem::CudaRenderSystem(
	V_Device& device, uint32_t nSwapChainImages, VkSemaphore cudaToVkSemaphore, VkSemaphore vkToCudaSemaphore,
	std::vector<V_GameObject>* objects, uint32_t h, uint32_t w, VkRenderPass renderPass) : height(h), width(w),
v_device(device), cudaUpdateVkSemaphore(cudaToVkSemaphore),
vkUpdateCudaSemaphore(vkToCudaSemaphore), gameObjects(objects), numSwapChainImages(nSwapChainImages) {

	c_createFunctions();
	c_createModel();

	checkCudaErrors(cudaStreamCreate(&streamToRun));
	c_importVkSemaphore();
	c_createSurfaceAndColorArray();
	c_createDescriptorSetLayout();
	c_createDescriptorPool();
	std::cout << descriptorPool << "\n";
	c_createPipelineLayout();
	c_createPipeline(renderPass);
	c_createImage();
	c_importImage();
	c_createImageView();
	c_createSampler();
	c_createDescriptorSets();
	c_updateDescriptorSets();

	samples = 1;

	uint32_t idealSquareSize = 50; // ???
	int tx = ceil(width / idealSquareSize);
	int ty = ceil(height / idealSquareSize);

	blocks = dim3(width / tx + 1, height / ty + 1);
	threads = dim3(tx, ty);

	checkCudaErrors(cudaMalloc((void**)&d_rand_state, width * height * sizeof(curandState)));
	cudaDeviceSetLimit(cudaLimitStackSize, 8192);

	int texnx, texny, texnn;
	tex_data = stbi_load("earthmap.jpg", &texnx, &texny, &texnn, 0);
	checkCudaErrors(cudaMalloc((void**)&d_tex_data, sizeof(unsigned char) * texnx * texny * 3));
	checkCudaErrors(cudaMemcpy(d_tex_data, tex_data, sizeof(unsigned char) * texnx * texny * 3, cudaMemcpyHostToDevice));

	LAUNCH_KERNEL(render_init, blocks, threads, 0, streamToRun, width, height, d_rand_state, 0);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&ranvec, sizeof(vec3*)));
	checkCudaErrors(cudaMalloc((void**)&perm_x, sizeof(int*)));
	checkCudaErrors(cudaMalloc((void**)&perm_y, sizeof(int*)));
	checkCudaErrors(cudaMalloc((void**)&perm_z, sizeof(int*)));

	checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));
	checkCudaErrors(cudaMalloc((void**)&d_list, numSpheres * sizeof(hitable*)));
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

	LAUNCH_KERNEL(create_world, 1, 1, 0, streamToRun, d_list, d_world, d_cam, numSpheres, width, height, ranvec, perm_x, perm_y, perm_z, d_rand_state, d_tex_data, texnx, texny);
}

CudaRenderSystem::~CudaRenderSystem() {
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
	//checkCudaErrors(cudaDestroyTextureObject(textureObjMipMapInput));
	vkDestroySampler(v_device.device(), sampler, nullptr);
	vkDestroyDescriptorSetLayout(v_device.device(), descriptorSetLayout, nullptr);
	vkDestroyDescriptorPool(v_device.device(), descriptorPool, nullptr);
	vkDestroyPipelineLayout(v_device.device(), pipelineLayout, nullptr);
	vkDestroyImageView(v_device.device(), imageView, nullptr);
	vkDestroyImage(v_device.device(), image, nullptr);
	vkFreeMemory(v_device.device(), imageMemory, nullptr);

	LAUNCH_KERNEL(free_world, 1, 1, 0, streamToRun, d_list, d_world, d_cam, numSpheres, ranvec, perm_x, perm_y, perm_z, d_tex_data);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_tex_data));
	checkCudaErrors(cudaFree(d_cam));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));


	cudaDeviceReset();
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
	LAUNCH_KERNEL(render<RGBA32>, blocks, threads, 0, streamToRun, d_surfaceObject, width, height, samples, d_cam, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	c_signalVkSemaphore(); // TODO: later signal semaphore async? 
	// c_updateDescriptorSets();
	v_pipeline->bindGraphics(commandBuffer);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);
	vkCmdDraw(commandBuffer, 6, 1, 0, 0);
}

__global__ void moveCamera(camera** d_cam, glm::vec3 position, glm::mat3 matrix) {
	(*d_cam)->updateCam(position, matrix);
}

void CudaRenderSystem::c_moveCamera(glm::vec3 position, glm::vec3 rotation) {
	if (!(position.length() > EPSILON) && !(rotation.length() > EPSILON)) return;
	const float c3 = glm::cos(rotation.z);
	const float s3 = glm::sin(rotation.z);
	const float c1 = glm::cos(rotation.x);
	const float s1 = glm::sin(rotation.x);
	const float c2 = glm::cos(rotation.y);
	const float s2 = glm::sin(rotation.y);

	// YXZ
	//const glm::vec3 u{(c1* c3 + s1 * s2 * s3), (c2* s3), (c1* s2* s3 - c3 * s1)};
	//const glm::vec3 v{(c3* s1* s2 - c1 * s3), (c2* c3), (c1* c3* s2 + s1 * s3)};
	//const glm::vec3 w{-(c2* s1), (s2), (c1* c2)};	

	// YX
	//const glm::vec3 u{c2, 0, s2};
	//const glm::vec3 v{s1*s2, c1, -s1*c2};
	//const glm::vec3 w{-(c1* s2), (s1), (c1* c2)};

	//// YZ
	//const glm::vec3 u{c2*c1, -s2, s2*c1};
	//const glm::vec3 v{s1 * c2, c1, s1 * s2};
	//const glm::vec3 w{-s2, 0, c2};

	// ZY
	//const glm::vec3 u{c2*c1, -s1*c2, s2};
	//const glm::vec3 v{s1, c1, 0};
	//const glm::vec3 w{-s2*c1, s1*s2, c2};

	// XY
	const glm::vec3 u{c2, s1*s2, s2*c1};
	const glm::vec3 v{0, c1, -s1};
	const glm::vec3 w{-s2, s1*c2, (c1* c2)};

	glm::mat3 viewMatrix = glm::mat3{ 1.f };
	// column row index
	viewMatrix[0][0] = u.x;
	viewMatrix[1][0] = u.y;
	viewMatrix[2][0] = u.z;
	viewMatrix[0][1] = v.x;
	viewMatrix[1][1] = v.y;
	viewMatrix[2][1] = v.z;
	viewMatrix[0][2] = w.x;
	viewMatrix[1][2] = w.y;
	viewMatrix[2][2] = w.z;
	//viewMatrix[0][3] = -glm::dot(u, position);
	//viewMatrix[1][3] = -glm::dot(v, position);
	//viewMatrix[2][3] = -glm::dot(w, position);

	LAUNCH_KERNEL(moveCamera, 1, 3, 0, streamToRun, d_cam, position, viewMatrix); // TODO: figure out blocks and threads
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
