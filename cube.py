import numpy as np
import matplotlib.pyplot as plt

class Plane:
    def __init__(self, point, normal, color):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.color = np.array(color, dtype=float)

    def intersect(self, ray_origin, ray_direction):
        denom = np.dot(ray_direction, self.normal)
        if np.abs(denom) < 1e-6:
            return None, None, None
        t = np.dot(self.point - ray_origin, self.normal) / denom
        if t < 0:
            return None, None, None
        return t, self.normal, ray_origin + ray_direction * t

class Cube:
    def __init__(self, center, side, color):
        self.center = np.array(center, dtype=float)
        self.side = side
        self.color = np.array(color, dtype=float)
        half = side / 2.0
        self.planes = [
            Plane(center + half * np.array([1, 0, 0]), np.array([1, 0, 0]), color),
            Plane(center + half * np.array([-1, 0, 0]), np.array([-1, 0, 0]), color),
            Plane(center + half * np.array([0, 1, 0]), np.array([0, 1, 0]), color),
            Plane(center + half * np.array([0, -1, 0]), np.array([0, -1, 0]), color),
            Plane(center + half * np.array([0, 0, 1]), np.array([0, 0, 1]), color),
            Plane(center + half * np.array([0, 0, -1]), np.array([0, 0, -1]), color),
        ]

    def intersect(self, ray_origin, ray_direction):
        min_dist = float('inf')
        hit_point = None
        normal = None
        for plane in self.planes:
            dist, n, p = plane.intersect(ray_origin, ray_direction)
            if dist is not None:
                # Check if the intersection is within the bounds of the cube face
                if all(abs(p - (self.center + n * self.side / 2)) <= self.side / 2):
                    if dist < min_dist:
                        min_dist = dist
                        hit_point = p
                        normal = n
        return min_dist if min_dist != float('inf') else None, normal

class Light:
    def __init__(self, position, intensity):
        self.position = np.array(position, dtype=float)
        self.intensity = intensity

def compute_lighting(point, normal, scene_light):
    light_dir = scene_light.position - point
    light_dir /= np.linalg.norm(light_dir)
    return max(np.dot(light_dir, normal), 0) * scene_light.intensity

def ray_from_camera(camera_position, x, y, image_width, image_height, fov):
    aspect_ratio = image_width / image_height
    scale = np.tan(fov * 0.5 * np.pi / 180)
    pixel_x = (2 * (x + 0.5) / image_width - 1) * aspect_ratio * scale
    pixel_y = (1 - 2 * (y + 0.5) / image_height) * scale
    ray_direction = np.array([pixel_x, pixel_y, -1], dtype=float)
    ray_direction /= np.linalg.norm(ray_direction)
    return camera_position, ray_direction

def render(scene, scene_light, camera_position, image_width, image_height, fov):
    image = np.zeros((image_height, image_width, 3), dtype=float)
    hit_count = 0  # Counter for debugging
    ambient_light = 0.1  # Ambient light intensity
    for y in range(image_height):
        for x in range(image_width):
            origin, direction = ray_from_camera(camera_position, x, y, image_width, image_height, fov)
            closest_obj = None
            min_dist = np.inf
            for obj in scene:
                dist, normal = obj.intersect(origin, direction)
                if dist is not None and dist < min_dist:
                    hit_count += 1  # Increment counter when a hit is detected
                    min_dist = dist
                    closest_obj = obj
                    hit_normal = normal
            if closest_obj:
                hit_point = origin + min_dist * direction
                light_intensity = compute_lighting(hit_point, hit_normal, scene_light) + ambient_light
                color = closest_obj.color * light_intensity
                image[y, x] = np.clip(color, 0, 1)
                #if np.any(image[y, x] > 0):
                    #print(f"Pixel color at ({x},{y}): {image[y, x]}")
    #print(f"Number of rays that hit the cube: {hit_count}")
    return image


# Scene setup
scene = [Cube([0, 0, -5], 2, [0, 1, 0])]  # Single green cube
scene_light = Light([-3, -4, -10], 1.5)  # Light setup
camera_position = np.array([22, -2, 0], dtype=float)  # Camera setup
image_width = 800
image_height = 600
fov = 90

# Render the scene
image = render(scene, scene_light, camera_position, image_width, image_height, fov)

# Display the result using matplotlib
plt.figure(figsize=(10, 7.5))
plt.imshow(image)
plt.axis('off')
plt.show()
