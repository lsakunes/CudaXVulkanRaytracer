#ifndef VGAMEOBJECTHPP
#define VGAMEOBJECTHPP

#include "v_model.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>

namespace v {

struct TransformComponent{
	glm::vec3 translation{};
	float scale = 1.f;
	glm::vec3 rotation{}; //TODO: check if we should use QUATERNIONS AAAAAAAAAAAAAAAAAA

	// translate * Ry * Rx * Rz * scale
	// tait-bryan angles YXZ
    glm::mat4 mat4() {
        const float c3 = glm::cos(rotation.z);
        const float s3 = glm::sin(rotation.z);
        const float c2 = glm::cos(rotation.x);
        const float s2 = glm::sin(rotation.x);
        const float c1 = glm::cos(rotation.y);
        const float s1 = glm::sin(rotation.y);
        return glm::mat4{
            {
                scale * (c1* c3 + s1 * s2 * s3),
                    scale * (c2* s3),
                    scale * (c1* s2* s3 - c3 * s1),
                    0.0f,
            },
        {
            scale * (c3 * s1 * s2 - c1 * s3),
            scale * (c2 * c3),
            scale * (c1 * c3 * s2 + s1 * s3),
            0.0f,
        },
        {
            scale * (c2 * s1),
            scale * (-s2),
            scale * (c1 * c2),
            0.0f,
        },
            { translation.x, translation.y, translation.z, 1.0f }};
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
	TransformComponent transform{};
	glm::vec3 color;


private:
	V_GameObject(id_t objId) : id(objId) {}

	id_t id;
};
}

#endif