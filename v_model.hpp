#ifndef VMODELHPP
#define VMODELHPP

#include "v_device.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <vector>

namespace v {

class V_Model {
public:

	// always remember to update VertexInputAttributeDescription in model.cpp
	struct Vertex { // TODO: restructuring memory to be multiple structs or smth might be faster
		glm::vec3 position;
		glm::vec3 color;

		static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
		static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
	};

	V_Model(V_Device &device, const std::vector<Vertex> &vertices);
	~V_Model();

	V_Model(const V_Model&) = delete;
	V_Model& operator=(const V_Model&) = delete;

	void bind(VkCommandBuffer commandBuffer);
	void draw(VkCommandBuffer commandBuffer);


private:
	void createVertexBuffers(const std::vector<Vertex>& vertices);

	V_Device& v_device;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	uint32_t vertexCount;
};
}
#endif