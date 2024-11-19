#include "simple_render_system.hpp"

namespace v{
void SimpleRenderSystem::createPipelineLayout() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SimplePushConstantData);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr; // TODO: textures and other buffers
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(v_device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
        VK_SUCCESS) {
        throw std::runtime_error("error at createPipelineLayoutInfo");
    }
}

SimpleRenderSystem::SimpleRenderSystem(V_Device& device, VkRenderPass renderPass) : v_device(device) {
    createPipelineLayout();
    createPipeline(renderPass);
}

void SimpleRenderSystem::createPipeline(VkRenderPass renderPass) {
    assert(pipelineLayout != nullptr && "Cannot create pipeline before pipline layout");

    PipelineConfigInfo pipelineConfig = PipelineConfigInfo{};
    V_Pipeline::defaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.renderPass = renderPass;
    pipelineConfig.pipelineLayout = pipelineLayout;

    v_pipeline = std::make_unique<V_Pipeline>(
        v_device,
        "C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanInterop/shaders/vert.spv",
        "C:/Users/senuk/source/repos/Raytracing/CUDA_Vulkan_Interop/CudaVulkanInterop/shaders/frag.spv",
        pipelineConfig);
}

SimpleRenderSystem::~SimpleRenderSystem() {
    vkDestroyPipelineLayout(v_device.device(), pipelineLayout, nullptr);
}

void SimpleRenderSystem::renderGameObjects(VkCommandBuffer commandBuffer, std::vector<V_GameObject>& gameObjects, const V_Camera& camera) {
    v_pipeline->bindGraphics(commandBuffer);

    auto projectionView = camera.getProjection() * camera.getView();

    for (auto& obj : gameObjects) {
        SimplePushConstantData push{};
        auto modelMatrix = obj.transform.mat4();
        push.transform = projectionView * modelMatrix; //TODO: move camera projection to uniform buffer to calculate on gpu
        push.modelMatrix = modelMatrix;

        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(SimplePushConstantData),
            &push);
        obj.model->bind(commandBuffer);
        obj.model->draw(commandBuffer);
    }
}
}