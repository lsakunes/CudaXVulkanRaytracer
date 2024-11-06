#ifndef SIMPLERENDERHPP
#define SIMPLERENDERHPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/constants.hpp>

#include "v_pipeline.hpp"
#include "v_gameobject.hpp"

namespace v {

// TODO: use for camera
struct SimplePushConstantData {  // size must be multiple of 4
    glm::mat4 transform{1.f};
    alignas(16) glm::vec3 color;
};

class SimpleRenderSystem {
public:

    SimpleRenderSystem(V_Device& device, VkRenderPass renderPass);
    ~SimpleRenderSystem();

    SimpleRenderSystem(const SimpleRenderSystem&) = delete;
    SimpleRenderSystem& operator=(const SimpleRenderSystem&) = delete;

    void renderGameObjects(VkCommandBuffer commandBuffer, std::vector<V_GameObject>& gameObjects);

private:
    void createPipelineLayout();
    void createPipeline(VkRenderPass renderPass);

    V_Device& v_device;

    std::unique_ptr<V_Pipeline> v_pipeline;
    VkPipelineLayout pipelineLayout;

};
}
// TODO: check max push constant size

#endif