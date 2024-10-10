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
#include "v_window.hpp"
#include "v_pipeline.hpp"
#include "v_device.hpp"
#include "v_swap_chain.hpp"

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
    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers() {}
    void drawFrame() {}

    V_Window v_window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    V_Device v_device{ v_window };
    V_SwapChain v_swapchain{ v_device, v_window.getExtent() };
    std::unique_ptr<V_Pipeline> v_pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkCommandBuffer> commandBuffers;

};

void Vulkan::run() {
    while (!v_window.shouldClose()) {
        glfwPollEvents();
    }
}

void Vulkan::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr; // textures and other buffers
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (vkCreatePipelineLayout(v_device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
        VK_SUCCESS) {
        throw std::runtime_error("error at createPipelineLayoutInfo");
    }
}

void Vulkan::createPipeline() {
    PipelineConfigInfo pipelineConfig = V_Pipeline::defaultPipelineConfigInfo(v_swapchain.width(), v_swapchain.height());

    std::cout << &pipelineConfig << "\n";
    std::cout << pipelineConfig.colorBlendInfo.pAttachments->colorWriteMask << "\n"; // 15      (correct)
    std::cout << pipelineConfig.colorBlendInfo.pAttachments->colorWriteMask << "\n"; // 32763   (???)
    //what the fuck is happening

    pipelineConfig.renderPass = v_swapchain.getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;

    v_pipeline = std::make_unique<V_Pipeline>(
        v_device,
        "./shaders/vert.spv",
        "./shaders/frag.spv",
        pipelineConfig);
}

Vulkan::Vulkan() {
    createPipelineLayout();
    createPipeline();
    createCommandBuffers();
}

Vulkan::~Vulkan() {
    vkDestroyPipelineLayout(v_device.device(), pipelineLayout, nullptr);
}
}
#endif