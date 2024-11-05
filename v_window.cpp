#include "v_window.hpp"

namespace v {
V_Window::~V_Window() {
	glfwDestroyWindow(window);
	glfwTerminate(); 
}

void V_Window::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void V_Window::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
	if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface");
	}
}

void V_Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	auto v_window = reinterpret_cast<V_Window *>(glfwGetWindowUserPointer(window));
	v_window->framebufferResized = true;
	v_window->width = width;
	v_window->height = height;
}
}