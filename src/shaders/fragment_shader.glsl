#version 330 core

in vec3 frag_pos;
in vec3 normal;
in vec3 light_dir;
in vec3 view_dir;
flat in int material_index;

out vec4 frag_color;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    vec3 La;
    vec3 Ld;
    vec3 Ls;
};

uniform Material materials[100];  // Assuming a maximum of 100 materials

void main() {
    vec3 norm = normalize(normal);
    Material mat = materials[material_index];

    // Ambient
    vec3 ambient = mat.La * mat.ambient;

    // Diffuse
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = mat.Ld * (diff * mat.diffuse);

    // Specular
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), mat.shininess);
    vec3 specular = mat.Ls * (spec * mat.specular);

    vec3 result = ambient + diffuse + specular;
    frag_color = vec4(result, 1.0);
}