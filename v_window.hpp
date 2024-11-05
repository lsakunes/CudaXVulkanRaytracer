#ifndef VWINDOWHPP
#define VWINDOWHPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>

namespace v {

class V_Window {
public:
	V_Window(int w, int h, std::string name) : width(w), height(h), windowName(name) { initWindow(); };
	~V_Window();
	
	V_Window(const V_Window&) = delete;
	V_Window& operator=(const V_Window&) = delete;

	bool shouldClose() { return glfwWindowShouldClose(window); }
	VkExtent2D getExtent() { return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) }; }

	void createWindowSurface(VkInstance insance, VkSurfaceKHR* surface);
	bool wasWindowResized() { return framebufferResized; }
	void resetWindowResizedFlag() { framebufferResized = false; }
private:
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
	void initWindow();

	int width;
	int height;
	bool framebufferResized;


	std::string windowName;
	GLFWwindow* window;

};
}


#endif