#include "v_model.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <unordered_map>

#include <iostream>
#include "v_utils.hpp"

namespace std {
template<>
struct hash<v::V_Model::Vertex> {
	size_t operator()(v::V_Model::Vertex const& vertex) const {
		size_t seed = 0;
		v::hashCombine(seed, vertex.position, vertex.color, vertex.normal, vertex.uv);
		return seed;
	}
};
}



namespace v {

V_Model::V_Model(V_Device& device, const V_Model::Builder &builder) : v_device{ device } {
	createVertexBuffers(builder.vertices);
	createIndexBuffers(builder.indices);
	createExternalVertexBuffer(builder.vertices);
}

V_Model::~V_Model() {
	// TODO: At some point we will have to use the Vulkan Memory Allocator for big scenes
	// Buffers and memory are handled separately because allocating memory takes time, and there's
	// a hard limit to the total number of allocations
	// 
	// Allocate bigger chunks and give parts to different resources 
	//
	vkDestroyBuffer(v_device.device(), vertexBuffer, nullptr);
	vkDestroyBuffer(v_device.device(), extVertexBuffer, nullptr);
	vkFreeMemory(v_device.device(), vertexBufferMemory, nullptr);
	vkFreeMemory(v_device.device(), extVertexBufferMemory, nullptr);
	if (hasIndexBuffer) {
		vkDestroyBuffer(v_device.device(), indexBuffer, nullptr);
		vkFreeMemory(v_device.device(), indexBufferMemory, nullptr);
	}
}


void V_Model::createExternalVertexBuffer(const std::vector<Vertex>& vertices) {
	vertexCount = static_cast<uint32_t>(vertices.size());
	assert(vertexCount >= 3 && "Vertex count must be > 3");
	VkDeviceSize bufferSize = sizeof(vertices[0]) * vertexCount;
	vertexBufferSize = bufferSize;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer,
		stagingBufferMemory);

	void* data;
	vkMapMemory(v_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(v_device.device(), stagingBufferMemory);

	VkExternalMemoryBufferCreateInfo extBufferCreateInfo{};
	extBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	extBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	extBufferCreateInfo.pNext = VK_NULL_HANDLE;

	v_device.createExtBuffer(
		bufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		extVertexBuffer,
		extVertexBufferMemory,
		extBufferCreateInfo);
	v_device.copyBuffer(stagingBuffer, extVertexBuffer, bufferSize);

	vkDestroyBuffer(v_device.device(), stagingBuffer, nullptr);
	vkFreeMemory(v_device.device(), stagingBufferMemory, nullptr);
}

VkBuffer V_Model::createExternalIndexBuffer(const std::vector<uint32_t>& indices) {
	return VkBuffer{};
}

std::unique_ptr<V_Model> V_Model::createModelFromFile(V_Device& device, const std::string& filepath) {
	Builder builder{};
	builder.loadModel(filepath);
	std::cout << "Vertex count:" << builder.vertices.size() << "\n";
	return std::make_unique<V_Model>(device, builder);
}

void V_Model::createVertexBuffers(const std::vector<Vertex>& vertices) {
	vertexCount = static_cast<uint32_t>(vertices.size());
	assert(vertexCount >= 3 && "Vertex count must be > 3");
	VkDeviceSize bufferSize = sizeof(vertices[0]) * vertexCount;
	vertexBufferSize = bufferSize;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer,
		stagingBufferMemory);

	void* data;
	vkMapMemory(v_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(v_device.device(), stagingBufferMemory);

	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		vertexBuffer,
		vertexBufferMemory);
	v_device.copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	vkDestroyBuffer(v_device.device(), stagingBuffer, nullptr);
	vkFreeMemory(v_device.device(), stagingBufferMemory, nullptr);
}

void V_Model::createIndexBuffers(const std::vector<uint32_t>& indices) {
	indexCount = static_cast<uint32_t>(indices.size());
	hasIndexBuffer = indexCount > 0;
	if (!hasIndexBuffer) return;

	VkDeviceSize bufferSize = sizeof(indices[0]) * indexCount;
	indexBufferSize = bufferSize;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer,
		stagingBufferMemory);

	void* data;
	vkMapMemory(v_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(v_device.device(), stagingBufferMemory);

	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		indexBuffer,
		indexBufferMemory);
	v_device.copyBuffer(stagingBuffer, indexBuffer, bufferSize);

	vkDestroyBuffer(v_device.device(), stagingBuffer, nullptr);
	vkFreeMemory(v_device.device(), stagingBufferMemory, nullptr);
}

void V_Model::bind(VkCommandBuffer commandBuffer) {
	VkBuffer buffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);
	if (hasIndexBuffer) {
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
	}
}
void V_Model::draw(VkCommandBuffer commandBuffer) {
	if (hasIndexBuffer) {
		vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
	} else
	{ 
		vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
	}
}

std::vector<VkVertexInputBindingDescription> V_Model::Vertex::getBindingDescriptions() {
	std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
	bindingDescriptions[0].binding = 0;
	bindingDescriptions[0].stride = sizeof(Vertex);
	bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	return bindingDescriptions;
}


std::vector<VkVertexInputAttributeDescription> V_Model::Vertex::getAttributeDescriptions() {
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};

	attributeDescriptions.push_back({0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)});
	attributeDescriptions.push_back({1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)});
	attributeDescriptions.push_back({2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)});
	attributeDescriptions.push_back({3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)});
	return attributeDescriptions;
}

void V_Model::Builder::loadModel(const std::string& filepath) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())) {
		throw std::runtime_error(warn + err);
	}

	vertices.clear();
	indices.clear();

	std::unordered_map<Vertex, uint32_t> uniqueVertices{};
	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex{};

			if (index.vertex_index >= 0) {
				vertex.position = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};
				vertex.color = {
					attrib.colors[3 * index.vertex_index + 0],
					attrib.colors[3 * index.vertex_index + 1],
					attrib.colors[3 * index.vertex_index + 2]
				};
			}

			if (index.normal_index >= 0) {
				vertex.normal = {
					attrib.normals[3 * index.normal_index + 0],
					attrib.normals[3 * index.normal_index + 1],
					attrib.normals[3 * index.normal_index + 2]
				};
			}
			if (index.texcoord_index >= 0) {
				vertex.uv = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					attrib.texcoords[2 * index.texcoord_index + 1]
				};
			}

			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vertex);
			}
			indices.push_back(uniqueVertices[vertex]);
		}
	}
}

}