#ifndef VPIPELINEHPP
#define VPIPELINEHPP

#include "v_device.hpp"
#include "v_model.hpp"

// std
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>

namespace v {

struct PipelineConfigInfo {
	VkPipelineViewportStateCreateInfo viewportInfo;
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
	VkPipelineRasterizationStateCreateInfo rasterizationInfo;
	VkPipelineMultisampleStateCreateInfo multisampleInfo;
	VkPipelineColorBlendAttachmentState colorBlendAttachment;
	VkPipelineColorBlendStateCreateInfo colorBlendInfo;
	VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
	std::vector<VkDynamicState> dynamicStateEnables;
	VkPipelineDynamicStateCreateInfo dynamicStateInfo;
	VkPipelineLayout pipelineLayout = nullptr;
	VkRenderPass renderPass = nullptr;
	uint32_t subpass = 0;
};

class V_Pipeline{
public:
	V_Pipeline(
		V_Device &device,
		const std::string& vertFilepath,
		const std::string& fragFilepath,
		const PipelineConfigInfo& configInfo);
	~V_Pipeline();

	V_Pipeline(const V_Pipeline&) = delete;
	void operator=(const V_Pipeline&) = delete;

	static void defaultPipelineConfigInfo(PipelineConfigInfo& configInfo);

	void bindGraphics(VkCommandBuffer buffer);
private:
	static std::vector<char> readFile(const std::string& filepath);
	void createGraphicsPipeline(
		const std::string& vertFilepath,
		const std::string& fragFilepath,
		const PipelineConfigInfo& configInfo);

	void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);

	V_Device& v_device;
	VkPipeline graphicsPipeline; 
	VkShaderModule vertShaderModule;
	VkShaderModule fragShaderModule;
};

}

#endif