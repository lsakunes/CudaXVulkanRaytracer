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
private:
	void initWindow();

	const int width;
	const int height;


	std::string windowName;
	GLFWwindow* window;

};

V_Window::~V_Window() {
	glfwDestroyWindow(window);
	glfwTerminate(); 
}

void V_Window::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
}

void V_Window::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
	if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface");
	}
}
}


#endif