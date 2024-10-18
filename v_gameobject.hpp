#ifndef VGAMEOBJECTHPP
#define VGAMEOBJECTHPP

#include "v_model.hpp"

#include <memory>

namespace v {

struct Transform2dComponent{
	glm::vec2 translation{};
	glm::vec2 scale{1.f, 1.f};
	float rotation;

	glm::mat2 mat2() {
		const float s = glm::sin(rotation);
		const float c = glm::cos(rotation);
		glm::mat2 rotMatrix({ c,s }, { -s, c });
		glm::mat2 scaleMat{{scale.x, 0}, { 0, scale.y }};
		return rotMatrix * scaleMat;
	}
};

class V_GameObject {
public:
	using id_t = unsigned int;

	static V_GameObject createGameObject() {
		static id_t currentId = 0;
		return V_GameObject(currentId++);
	}

	V_GameObject(const V_GameObject&) = delete;
	V_GameObject& operator=(const V_GameObject&) = delete;
	V_GameObject(V_GameObject&&) = default;
	V_GameObject& operator=(V_GameObject&&) = default;

	const id_t getid() { return id; }

	std::shared_ptr<V_Model> model{};
	Transform2dComponent transform2d{};
	glm::vec3 color;


private:
	V_GameObject(id_t objId) : id(objId) {}

	id_t id;
};
}

#endif