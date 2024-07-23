# obj_loader.py
# Load an OBJ file into a numpy array for rendering with OpenGL

import numpy as np
import os
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
from collections import defaultdict

def random_color() -> Tuple[float, float, float]:
    return (random.random(), random.random(), random.random())

@dataclass
class Material:
    id: int = 0
    name: str = 'default'
    ambient: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    diffuse: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    specular: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    shininess: float = 50.0
    La: Tuple[float, float, float] = field(default_factory=random_color)
    Ld: Tuple[float, float, float] = field(default_factory=random_color)
    Ls: Tuple[float, float, float] = field(default_factory=random_color)

@dataclass
class Vertex:
    id: int
    x: float
    y: float
    z: float

@dataclass
class Normal:
    id: int
    x: float
    y: float
    z: float

@dataclass
class Face:
    id: int
    vertex_ids: List[int]
    normal_ids: List[int]
    material: Material

@dataclass
class ParsedObject:
    name: str
    face_count: int = 0
    materials: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_face(self, material_name: str):
        self.face_count += 1
        self.materials[material_name] += 1

    def print_details(self):
        print(f"Object: {self.name}")
        print(f"  Total faces: {self.face_count}")
        print("  Materials used:")
        for material, count in self.materials.items():
            print(f"    {material}: {count} faces")

@dataclass
class Mesh:
    vertices: List[Vertex]
    normals: List[Normal]
    faces: List[Face]
    materials: Dict[str, Material]
    parsed_objects: List[ParsedObject] = field(default_factory=list)

    def print_material_stats(self):
        material_face_count = defaultdict(int)
        for face in self.faces:
            material_face_count[face.material.name] += 1
        
        print("Total faces per material in the mesh:")
        for material, count in material_face_count.items():
            print(f"  {material}: {count} faces")

def load_mtl(filename: str) -> Dict[str, Material]:
    materials = {}
    current_material = None
    material_id = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('newmtl '):
                name = line.split()[1]
                material_id += 1
                current_material = Material(id=material_id, name=name)
                materials[name] = current_material
            elif line.startswith('Ka '):
                current_material.ambient = tuple(map(float, line.split()[1:4]))
            elif line.startswith('Kd '):
                current_material.diffuse = tuple(map(float, line.split()[1:4]))
            elif line.startswith('Ks '):
                current_material.specular = tuple(map(float, line.split()[1:4]))
            elif line.startswith('Ns '):
                current_material.shininess = float(line.split()[1])
    return materials

def calculate_normal(v1, v2, v3):
    u = np.subtract(v2, v1)
    v = np.subtract(v3, v1)
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else normal

def create_mesh(file_path: str) -> Mesh:
    vertices = []
    normals = []
    faces = []
    materials = {}
    parsed_objects = []
    current_material = None
    current_object = None
    vertex_id = 0
    normal_id = 0
    face_id = 0

    # Load materials if present
    dirname = os.path.dirname(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('mtllib '):
                mtl_path = os.path.join(dirname, line.strip().split()[1])
                if os.path.exists(mtl_path):
                    materials = load_mtl(mtl_path)
                else:
                    print(f"Warning: MTL file {mtl_path} not found. Using default material.")
                break

    # Ensure there's always a default material
    if 'default' not in materials:
        materials['default'] = Material(id=0, name='default')

    # Set the initial current_material to the default
    current_material = materials['default']

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('o '):
                obj_name = line.strip().split(maxsplit=1)[1]
                current_object = ParsedObject(name=obj_name)
                parsed_objects.append(current_object)
            elif line.startswith('v '):
                x, y, z = map(float, line.strip().split()[1:4])
                vertices.append(Vertex(id=vertex_id, x=x, y=y, z=z))
                vertex_id += 1
            elif line.startswith('vn '):
                x, y, z = map(float, line.strip().split()[1:4])
                normals.append(Normal(id=normal_id, x=x, y=y, z=z))
                normal_id += 1
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                vertex_ids = []
                normal_ids = []
                for part in parts:
                    v, _, n = (part.split('/') + ['', ''])[:3]
                    vertex_ids.append(int(v) - 1)
                    if n:
                        normal_ids.append(int(n) - 1)
                faces.append(Face(id=face_id, vertex_ids=vertex_ids, normal_ids=normal_ids, material=current_material))
                face_id += 1
                if current_object:
                    current_object.add_face(current_material.name)
            elif line.startswith('usemtl '):
                material_name = line.strip().split()[1]
                current_material = materials.get(material_name, materials['default'])

    # If no normals were provided, calculate them
    if not normals:
        normals = [Normal(id=i, x=0, y=0, z=0) for i in range(len(vertices))]
        for face in faces:
            v1, v2, v3 = [vertices[i] for i in face.vertex_ids[:3]]
            normal = calculate_normal(
                np.array([v1.x, v1.y, v1.z]),
                np.array([v2.x, v2.y, v2.z]),
                np.array([v3.x, v3.y, v3.z])
            )
            for vertex_id in face.vertex_ids:
                normals[vertex_id].x += normal[0]
                normals[vertex_id].y += normal[1]
                normals[vertex_id].z += normal[2]

        # Normalize the normals
        for normal in normals:
            magnitude = np.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
            if magnitude != 0:
                normal.x /= magnitude
                normal.y /= magnitude
                normal.z /= magnitude

    # Ensure each face has normal IDs and a material
    for face in faces:
        if not face.normal_ids:
            face.normal_ids = face.vertex_ids
        if face.material is None:
            face.material = materials['default']
    
    print(f"Loaded mesh with {len(vertices)} vertices, {len(normals)} normals, {len(faces)} faces, and {len(materials)} materials.")
    return Mesh(vertices=vertices, normals=normals, faces=faces, materials=materials, parsed_objects=parsed_objects)

def mesh_to_arrays(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Material], np.ndarray]:
    vertices = np.array([(v.x, v.y, v.z) for v in mesh.vertices], dtype=np.float32)
    normals = np.array([(n.x, n.y, n.z) for n in mesh.normals], dtype=np.float32)
    
    vertex_indices = []
    material_indices = []
    material_list = list(mesh.materials.values())
    material_dict = {m.name: i for i, m in enumerate(material_list)}
    
    for face in mesh.faces:
        vertex_indices.extend(face.vertex_ids)
        material_index = material_dict[face.material.name]
        material_indices.extend([material_index] * len(face.vertex_ids))
    
    vertex_indices = np.array(vertex_indices, dtype=np.uint32)
    material_indices = np.array(material_indices, dtype=np.int32)

    print(f"Vertices: {vertices.shape}, Normals: {normals.shape}, Vertex Indices: {vertex_indices.shape}, Materials: {len(mesh.materials)}, Material Indices: {material_indices.shape}")
    
    return vertices, normals, vertex_indices, material_list, material_indices

def load_obj(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Material], np.ndarray]:
    mesh = create_mesh(file_path)
    
    # # Print details for each parsed object
    # for obj in mesh.parsed_objects:
    #     obj.print_details()
    
    # # Print total material stats for the mesh
    # mesh.print_material_stats()
    
    return mesh_to_arrays(mesh)