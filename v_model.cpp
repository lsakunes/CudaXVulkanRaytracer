#include "v_model.hpp"

namespace v {

V_Model::V_Model(V_Device& device, const std::vector<Vertex>& vertices) : v_device{ device } {
	createVertexBuffers(vertices);
}

V_Model::~V_Model() {
	// TODO: At some point we will have to use the Vulkan Memory Allocator for big scenes
	// Buffers and memory are handled separately because allocating memory takes time, and there's
	// a hard limit to the total number of allocations
	// 
	// Allocate bigger chunks and give parts to different resources 
	//
	vkDestroyBuffer(v_device.device(), vertexBuffer, nullptr);
	vkFreeMemory(v_device.device(), vertexBufferMemory, nullptr);
}

void V_Model::createVertexBuffers(const std::vector<Vertex>& vertices) {
	vertexCount = static_cast<uint32_t>(vertices.size());
	assert(vertexCount >= 3 && "Vertex count must be > 3");
	VkDeviceSize bufferSize = sizeof(vertices[0]) * vertexCount;

	v_device.createBuffer(
		bufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		vertexBuffer,
		vertexBufferMemory);

	void* data;
	vkMapMemory(v_device.device(), vertexBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(v_device.device(), vertexBufferMemory);
}

void V_Model::bind(VkCommandBuffer commandBuffer) {
	VkBuffer buffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);
}
void V_Model::draw(VkCommandBuffer commandBuffer) {
	vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
}

std::vector<VkVertexInputBindingDescription> V_Model::Vertex::getBindingDescriptions() {
	std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
	bindingDescriptions[0].binding = 0;
	bindingDescriptions[0].stride = sizeof(Vertex);
	bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	return bindingDescriptions;
}


std::vector<VkVertexInputAttributeDescription> V_Model::Vertex::getAttributeDescriptions() {
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
	attributeDescriptions[0].binding = 0;
	attributeDescriptions[0].location = 0;
	attributeDescriptions[0].offset = offsetof(Vertex, position);
	attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
	attributeDescriptions[1].binding = 0;
	attributeDescriptions[1].location = 1;
	attributeDescriptions[1].offset = offsetof(Vertex, color);
	attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	return attributeDescriptions;
}

}