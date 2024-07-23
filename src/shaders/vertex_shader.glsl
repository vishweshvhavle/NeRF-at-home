#version 330 core

in vec3 vVertex;
in vec3 vNormal;
in int vMaterialIndex;

uniform mat4 vModel;
uniform mat4 vView;
uniform mat4 vProjection;
uniform vec3 light_pos;
uniform vec3 eye_pos;

out vec3 frag_pos;
out vec3 normal;
out vec3 light_dir;
out vec3 view_dir;
flat out int material_index;

void main() {
    vec4 world_pos = vModel * vec4(vVertex, 1.0);
    frag_pos = vec3(world_pos);
    gl_Position = vProjection * vView * world_pos;
    normal = mat3(transpose(inverse(vModel))) * vNormal;
    light_dir = normalize(light_pos - frag_pos);
    view_dir = normalize(eye_pos - frag_pos);
    material_index = vMaterialIndex;
}