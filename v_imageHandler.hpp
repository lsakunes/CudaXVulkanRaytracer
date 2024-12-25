#include "v_device.hpp"


namespace v {
	class V_ImageHandler {
	public:
		V_ImageHandler(V_Device& device) : v_device(device) {
		};
		~V_ImageHandler() {
			vkDestroySampler(v_device.device(), textureSampler, nullptr);
			vkDestroyImageView(v_device.device(), textureImageView, nullptr);
			vkDestroyImage(v_device.device(), textureImage, nullptr);
			vkFreeMemory(v_device.device(), textureImageMemory, nullptr);

			vkDestroyDescriptorSetLayout(v_device.device(), descriptorSetLayout, nullptr);
		}

		void createTextureImage();
		void createTextureImageView();
		void createTextureSampler();
		void createDescriptorSetLayout();
		void createDescriptorSets();




		VkImage textureImage;
		VkDeviceMemory textureImageMemory;
		VkImageView textureImageView;
		VkSampler textureSampler;
		VkDescriptorSetLayout descriptorSetLayout;
		V_Device& v_device;
		unsigned int imageWidth, imageHeight;
	};
}