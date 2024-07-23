# data_generator.py 
# Script to render frames of a 3D model along a viewing circle and save the frames along with pose data  

import argparse
import glfw
from OpenGL.GL import *
import numpy as np
from obj_loader import load_obj
from shader_loader import load_shaders
import glm
import imgui
from imgui.integrations.glfw import GlfwRenderer
import cv2
import os
import json
from datetime import datetime
from tqdm import tqdm

# Global variables
rotation_matrix = glm.mat4(1)
last_x, last_y = 400, 400
left_mouse_button_pressed = False
theta = 0.0
phi = 0.0
radius = 5.0
shader_params = {
    "light_pos": [100.2, 120.0, 2.0],
    "eye_pos": [0.0, 0.0, 5.0]
}
gaussian_blur_params = {
    "kernel_size": (5, 5),
    "sigma": 0
}

def init_glfw():
    # print("Initializing GLFW...")
    if not glfw.init():
        raise Exception("GLFW can't be initialized")
    window = glfw.create_window(1200, 800, "OBJ Viewer", None, None)
    glfw.set_window_pos(window, 100, 100)
    glfw.make_context_current(window)
    return window

def setup_callbacks(window):
    # print("Setting up callbacks...")
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

def mouse_button_callback(window, button, action, mods):
    global left_mouse_button_pressed, last_x, last_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        left_mouse_button_pressed = action == glfw.PRESS
        last_x, last_y = glfw.get_cursor_pos(window)

def cursor_position_callback(window, xpos, ypos):
    global rotation_matrix, last_x, last_y
    if left_mouse_button_pressed:
        dx, dy = xpos - last_x, ypos - last_y
        sensitivity = 0.01
        rotation_x = glm.rotate(glm.mat4(1), sensitivity * dy, glm.vec3(1, 0, 0))
        rotation_y = glm.rotate(glm.mat4(1), sensitivity * dx, glm.vec3(0, 1, 0))
        rotation_matrix = rotation_y * rotation_x * rotation_matrix
        last_x, last_y = xpos, ypos

def load_model(model):
    print("Loading model...")
    return load_obj(model)

def setup_buffers(vertices, normals, vertex_indices, material_indices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Vertex buffer
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Normal buffer
    nbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Material index buffer
    mbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, mbo)
    glBufferData(GL_ARRAY_BUFFER, material_indices.nbytes, material_indices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribIPointer(2, 1, GL_INT, 0, None)

    # Element buffer
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertex_indices.nbytes, vertex_indices, GL_STATIC_DRAW)

    glBindVertexArray(0)
    return vao, len(vertex_indices)

def setup_materials(shader_program, materials):
    for i, material in enumerate(materials):
        base_name = f"materials[{i}]"
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.ambient"), 1, material.ambient)
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.diffuse"), 1, material.diffuse)
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.specular"), 1, material.specular)
        glUniform1f(glGetUniformLocation(shader_program, f"{base_name}.shininess"), material.shininess)
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.La"), 1, material.La)
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.Ld"), 1, material.Ld)
        glUniform3fv(glGetUniformLocation(shader_program, f"{base_name}.Ls"), 1, material.Ls)

def setup_shaders():
    # print("Setting up shaders...")
    shader_program = load_shaders("src/shaders/vertex_shader.glsl", "src/shaders/fragment_shader.glsl")
    glUseProgram(shader_program)
    return shader_program

def setup_matrices(shader_program, scale, axis, angle):
    # print("Setting up matrices...")
    model = glm.scale(glm.mat4(1.0), glm.vec3(scale, scale, scale))
    model = glm.rotate(model, glm.radians(angle), glm.vec3(*axis))
    view = glm.lookAt(glm.vec3(*shader_params["eye_pos"]), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 1200 / 800, 0.1, 100.0)

    model_loc = glGetUniformLocation(shader_program, "vModel")
    view_loc = glGetUniformLocation(shader_program, "vView")
    projection_loc = glGetUniformLocation(shader_program, "vProjection")
    light_pos_loc = glGetUniformLocation(shader_program, "light_pos")
    eye_pos_loc = glGetUniformLocation(shader_program, "eye_pos")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniform3fv(light_pos_loc, 1, shader_params["light_pos"])

    return model, view, projection, model_loc, view_loc, projection_loc, eye_pos_loc

def render_frame(vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, eye_pos_loc, light_pos_loc):
    glClearColor(1.0, 1.0, 1.0, 1.0)  # Set background to white
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    view = view * rotation_matrix
    
    glUseProgram(shader_program)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniform3fv(eye_pos_loc, 1, shader_params["eye_pos"])
    glUniform3fv(light_pos_loc, 1, shader_params["light_pos"])
    
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def render_video(window, vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, eye_pos_loc, light_pos_loc, output_dir, frame_count, obj_model_name):
    print("Rendering frames and saving pose data...")
    
    # Create directory for output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"{obj_model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    output_filename = f"video_{obj_model_name}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 24, (1200, 800))
    
    pose_data = []
    
    for i in tqdm(range(frame_count), desc="Rendering frames"):
        angle = glm.radians(i * (360 / frame_count))
        x = radius * glm.sin(angle) * glm.cos(glm.radians(phi))
        y = radius * glm.sin(glm.radians(phi))
        z = radius * glm.cos(angle) * glm.cos(glm.radians(phi))
        
        shader_params["eye_pos"] = [x, y, z]
        view = glm.lookAt(glm.vec3(x, y, z), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        
        render_frame(vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, eye_pos_loc, light_pos_loc)
        
        # Capture frame
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, 1200, 800, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(800, 1200, 3)
        image = cv2.flip(image, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply Gaussian blur
        blurred_image = apply_gaussian_blur(image, gaussian_blur_params["kernel_size"], gaussian_blur_params["sigma"])
        
        # Save frame
        frame_filename = f"frame_{i:04d}.png"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, blurred_image)
        out.write(blurred_image)
        
        # Collect pose data
        pose_data.append({
            "frame": i,
            "filename": f"frames/{frame_filename}",
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "direction": {"theta": float(i * (360 / frame_count)), "phi": float(phi)}
        })
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Save pose data
    with open(os.path.join(output_dir, "pose.json"), "w") as f:
        json.dump(pose_data, f, indent=2)

    out.release()
    print(f"Video saved to {output_path}")
    print(f"Frames and pose data saved in {output_dir}")

def main(args):
    global theta, phi, radius, gaussian_blur_params
    window = init_glfw()
    setup_callbacks(window)
    
    vertices, normals, vertex_indices, materials, material_indices = load_obj(args.obj_model)
    vao, num_indices = setup_buffers(vertices, normals, vertex_indices, material_indices)
    
    shader_program = setup_shaders()
    model, view, projection, model_loc, view_loc, projection_loc, eye_pos_loc = setup_matrices(shader_program, args.scale, args.axis, args.angle)
    light_pos_loc = glGetUniformLocation(shader_program, "light_pos")
    
    setup_materials(shader_program, materials)
    glEnable(GL_DEPTH_TEST)
    
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    render_video_flag = False
    
    # Extract obj_model_name from the input path
    obj_model_name = os.path.splitext(os.path.basename(args.obj_model))[0]
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        imgui.begin("Viewing Circle Parameters")
        changed, value = imgui.slider_float("Theta", theta, -180.0, 180.0)
        if changed:
            theta = value
        changed, value = imgui.slider_float("Phi", phi, -90.0, 90.0)
        if changed:
            phi = value
        changed, value = imgui.slider_float("Radius", radius, 1.0, 10.0)
        if changed:
            radius = value
        
        imgui.text("Light Position")
        changed, value = imgui.slider_float3("Light Pos", *shader_params["light_pos"], -200.0, 200.0)
        if changed:
            shader_params["light_pos"] = list(value)
        
        imgui.text("Gaussian Blur Parameters")
        changed, value = imgui.slider_int("Kernel Size", gaussian_blur_params["kernel_size"][0], 1, 15, format="%d")
        if changed:
            gaussian_blur_params["kernel_size"] = (value, value)
        changed, value = imgui.slider_float("Sigma", gaussian_blur_params["sigma"], 0.0, 10.0)
        if changed:
            gaussian_blur_params["sigma"] = value
        
        changed, value = imgui.input_text("Output Directory", args.output_dir, 256)
        if changed:
            args.output_dir = value
        changed, value = imgui.slider_int("Frame Count", args.frame_count, 1, 360)
        if changed:
            args.frame_count = value
        
        if imgui.button("Render"):
            render_video_flag = True
        imgui.end()
        
        if render_video_flag:
            render_video(window, vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, eye_pos_loc, light_pos_loc, args.output_dir, args.frame_count, obj_model_name)
            render_video_flag = False
        else:
            x = radius * glm.sin(glm.radians(theta)) * glm.cos(glm.radians(phi))
            y = radius * glm.sin(glm.radians(phi))
            z = radius * glm.cos(glm.radians(theta)) * glm.cos(glm.radians(phi))
            shader_params["eye_pos"] = [x, y, z]
            view = glm.lookAt(glm.vec3(x, y, z), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
            render_frame(vao, num_indices, shader_program, model, view, projection, model_loc, view_loc, eye_pos_loc, light_pos_loc)
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
    
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRF-at-home: Bunny Model Renderer")
    parser.add_argument("--obj_model", type=str, default="src/assets/bunny.obj", help="Path to the OBJ model file")
    parser.add_argument("--axis", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Axis to rotate the model along")
    parser.add_argument("--angle", type=float, default=0.0, help="Angle to rotate the model")
    parser.add_argument("--scale", type=float, default=0.005, help="Scale for OBJ model file")
    parser.add_argument("--theta", type=float, default=0.0, help="Initial theta angle for viewing circle")
    parser.add_argument("--phi", type=float, default=0.0, help="Initial phi angle for viewing circle")
    parser.add_argument("--radius", type=float, default=5.0, help="Initial radius for viewing circle")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=float, default=0.0, help="Sigma value for Gaussian blur")
    parser.add_argument("--output_dir", type=str, default="src/data", help="Output directory for rendered frames with poses")
    parser.add_argument("--frame_count", type=int, default=10, help="Number of frames to render")
    parser.add_argument("--vertex_shader", type=str, default="src/shaders/vertex_shader.glsl", help="Path to vertex shader file")
    parser.add_argument("--fragment_shader", type=str, default="src/shaders/fragment_shader.glsl", help="Path to fragment shader file")
    args = parser.parse_args()
    
    # Update global variables with command-line arguments
    theta = args.theta
    phi = args.phi
    radius = args.radius
    gaussian_blur_params["kernel_size"] = (args.kernel_size, args.kernel_size)
    gaussian_blur_params["sigma"] = args.sigma
    
    main(args)
