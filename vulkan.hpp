#ifndef VULKANCPP
#define VULKANCPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>
#include <vector>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <array>

#include "v_window.hpp"
#include "v_pipeline.hpp"
#include "v_device.hpp"
#include "v_swap_chain.hpp"
#include "v_model.hpp"
#include "v_gameobject.hpp"

namespace v{

// TODO: use for camera
struct SimplePushConstantData {  // size must be multiple of 4
    glm::mat2 transform {1.f};
    alignas(16) glm::mat2 objTransform {1.f};
    alignas(16) glm::vec3 color;
};

class Vulkan {
public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    Vulkan();
    ~Vulkan();

    Vulkan(const Vulkan&) = delete;
    Vulkan& operator=(const Vulkan&) = delete;

    void run();
private:
    void loadGameObjects();
    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers();
    void drawFrame();
    void renderGameObjects(VkCommandBuffer commandBuffer);
    void recordCommandBuffer(int imageIndex);

    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    std::unique_ptr <V_SwapChain> v_swapchain;
    std::unique_ptr<V_Pipeline> v_pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<V_GameObject> gameObjects;

};

void Vulkan::run() {
    while (!v_window.shouldClose()) {
        glfwPollEvents();
        drawFrame();
    }
    
    vkDeviceWaitIdle(v_device.device());
}

void Vulkan::createPipelineLayout() {

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
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

void Vulkan::createPipeline() {
    PipelineConfigInfo pipelineConfig = PipelineConfigInfo{};
    V_Pipeline::defaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.renderPass = v_swapchain->getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;

    v_pipeline = std::make_unique<V_Pipeline>(
        v_device,
        "./shaders/vert.spv",
        "./shaders/frag.spv",
        pipelineConfig);
}

Vulkan::Vulkan() {
    loadGameObjects();
    createPipelineLayout();
    createPipeline();
    createCommandBuffers();
}

Vulkan::~Vulkan() {
    vkDestroyPipelineLayout(v_device.device(), pipelineLayout, nullptr);
}

void Vulkan::createCommandBuffers() {
    commandBuffers.resize(v_swapchain->imageCount());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = v_device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(v_device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Error allocating command buffers");
    }
}

void Vulkan::recordCommandBuffer(int imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to being recording command buffer");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = v_swapchain.getRenderPass();
    renderPassInfo.framebuffer = v_swapchain.getFrameBuffer(imageIndex);

    renderPassInfo.renderArea.offset = { 0,0 };
    renderPassInfo.renderArea.extent = v_swapchain.getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0, 0, 0, 1.0f };
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    VkExtent2D extent = v_swapchain->getSwapChainExtent();
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    VkRect2D scissor{ {0,0}, v_swapchain->GetSwapChainExtent() };
    vkCmdSetViewport(commandBuffers[imageIndex], 0, 1, &viewport);
    vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &scissor);



    vkCmdBeginRenderPass(commandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // TODO: contents inline might change to use secondary command buffers with cuda?? maybe??

    renderGameObjects(commandBuffers[imageIndex]);

    vkCmdEndRenderPass(commandBuffers[imageIndex]);
    if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer");
    }
}


void Vulkan::drawFrame() {
    uint32_t imageIndex;
    auto result = v_swapchain.acquireNextImage(&imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    recordCommandBuffer(imageIndex);
    result = v_swapchain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || v_window.wasWindowResized()) {
        v_window.resetWindowResizedFlag();
        recreateSwapChain();
        return;
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image");
    }
}

void Vulkan::renderGameObjects(VkCommandBuffer commandBuffer) {
    v_pipeline->bindGraphics(commandBuffer);

    for (auto& obj : gameObjects) {
        obj.transform2d.rotation = glm::mod(obj.transform2d.rotation + 0.1f, glm::two_pi<float>());
        SimplePushConstantData push{};
        push.objTransform = obj.transform2d.mat2();
        push.color = obj.color;
        // TODO: camera matrix
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(SimplePushConstantData),
            &push);
        obj.model->bind(commandBuffer);
        obj.model->draw(commandBuffer);
    }

}

void Vulkan::loadGameObjects() {
    std::vector<V_Model::Vertex> vertices {
        {{0.0f, -0.5f}, { 1.0f, 0, 1.0f }},
        { {0.5f, 0.5f}, {1.0f, 1.0f, 0} },
        { {-0.5, 0.5f},{0, 1.0f, 1.0f} }
    };
    auto v_model = std::make_shared<V_Model>(v_device, vertices);

    auto triangle = V_GameObject::createGameObject();
    triangle.model = v_model;
    triangle.color = { 0.f, 1.f, .5f };
    triangle.transform2d.translation.x = .2f;
    triangle.transform2d.scale = { 2.f, .5f };
    triangle.transform2d.rotation = .25f * glm::two_pi<float>();

    gameObjects.push_back(std::move(triangle));
}

}
#endif

// TODO: check max push constant size