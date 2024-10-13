#ifndef VULKANCPP
#define VULKANCPP
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
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


namespace v{
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
    void loadModels();
    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers();
    void drawFrame();

    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    V_SwapChain v_swapchain{ v_device, v_window.getExtent() };
    std::unique_ptr<V_Pipeline> v_pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkCommandBuffer> commandBuffers;
    std::unique_ptr<V_Model> v_model;

};

void Vulkan::run() {
    while (!v_window.shouldClose()) {
        glfwPollEvents();
        drawFrame();
    }
    
    vkDeviceWaitIdle(v_device.device());
}

void Vulkan::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr; // TODO: textures and other buffers
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (vkCreatePipelineLayout(v_device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
        VK_SUCCESS) {
        throw std::runtime_error("error at createPipelineLayoutInfo");
    }
}

void Vulkan::createPipeline() {
    PipelineConfigInfo pipelineConfig = PipelineConfigInfo{};
    V_Pipeline::defaultPipelineConfigInfoX(pipelineConfig, v_swapchain.width(), v_swapchain.height());

    pipelineConfig.renderPass = v_swapchain.getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;

    v_pipeline = std::make_unique<V_Pipeline>(
        v_device,
        "./shaders/vert.spv",
        "./shaders/frag.spv",
        pipelineConfig);
}

Vulkan::Vulkan() {
    loadModels();
    createPipelineLayout();
    createPipeline();
    createCommandBuffers();
}

Vulkan::~Vulkan() {
    vkDestroyPipelineLayout(v_device.device(), pipelineLayout, nullptr);
}

void Vulkan::createCommandBuffers() {
    commandBuffers.resize(v_swapchain.imageCount());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = v_device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(v_device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Error allocating command buffers");
    }

    for (int i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to being recording command buffer");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = v_swapchain.getRenderPass();
        renderPassInfo.framebuffer = v_swapchain.getFrameBuffer(i);

        renderPassInfo.renderArea.offset = { 0,0 };
        renderPassInfo.renderArea.extent = v_swapchain.getSwapChainExtent();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { 0, 0, 0, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // TODO: contents inline might change to use secondary command buffers with cuda?? maybe??

        v_pipeline->bindGraphics(commandBuffers[i]);
        v_model->bind(commandBuffers[i]);
        v_model->draw(commandBuffers[i]);

        vkCmdEndRenderPass(commandBuffers[i]);
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer");
        }
    }
}



void Vulkan::drawFrame() {
    uint32_t imageIndex;
    auto result = v_swapchain.acquireNextImage(&imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    result = v_swapchain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image");
    }
}

void Vulkan::loadModels() {
    std::vector<V_Model::Vertex> vertices {
        {{0.0f, -0.5f}, { 1.0f, 0, 1.0f }},
        { {0.5f, 0.5f}, {1.0f, 1.0f, 0} },
        { {-0.5, 0.5f},{0, 1.0f, 1.0f} }
    };

    v_model = std::make_unique<V_Model>(v_device, vertices);
}

}
#endif