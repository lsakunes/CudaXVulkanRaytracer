#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    mat2 transform;
    mat2 objTransform;
    vec3 color;
} push;

void main() {
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
    //outColor = vec4(fragColor, 1.0);
}