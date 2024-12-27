#version 450
layout(set = 0, binding = 0) uniform sampler2D renderedImage;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(renderedImage, fragUV);// + vec4(fragUV.x/2, fragUV.y/2, 0, 1);
}
