import bpy
import os
import random
import time
import math
from glob import glob
from mathutils import Vector


class RainPBRGenerator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.scene = bpy.context.scene
        self.setup_render_engine()

        self.path_blended = os.path.join(base_dir, "raindrop", "syn", "blended")
        self.path_clean = os.path.join(base_dir, "raindrop", "syn", "clean")
        for p in [self.path_blended, self.path_clean]:
            os.makedirs(p, exist_ok=True)

    def setup_render_engine(self):
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = 128
        self.scene.cycles.use_denoising = True

    def get_scene_objects(self):
        self.bg_plane = bpy.data.objects.get("Background")
        self.glass_obj = bpy.data.objects.get("GlassPlane")
        self.cam = bpy.data.objects.get("Camera")
        self.glass_nodes = self.glass_obj.active_material.node_tree.nodes

        # 预留节点变量获取，以防后续需要用到
        self.raindrop_bsdf = self.glass_nodes.get("Principled BSDF")
        self.glass_base_bsdf = self.glass_nodes.get("Principled BSDF.001")
        self.bump_node = self.glass_nodes.get("Bump")
        self.value_node = self.glass_nodes.get("值")
        self.mix_node = self.glass_nodes.get("混合着色器")

    def calculate_view_size(self, distance):
        cam_data = self.cam.data
        cam_data.sensor_fit = 'VERTICAL'
        aspect_ratio = self.scene.render.resolution_x / self.scene.render.resolution_y

        v_fov_half = math.atan(cam_data.sensor_height / (2 * cam_data.lens))
        half_height = math.tan(v_fov_half) * distance
        half_width = half_height * aspect_ratio

        return half_width * 1.15, half_height * 1.15

    def setup_background(self, distance):
        w, h = self.calculate_view_size(distance)
        obj = self.bg_plane
        obj.location = (0, self.cam.location.y + distance, 0)
        obj.rotation_euler = (math.radians(90), 0, 0)
        obj.scale[0] = w
        obj.scale[1] = h
        obj.scale[2] = 1.0

    def setup_glass(self, min_dist=1.0, max_dist=1.1):
        dist_to_cam = random.uniform(min_dist, max_dist)
        w, h = self.calculate_view_size(dist_to_cam)
        obj = self.glass_obj
        obj.location.y = self.cam.location.y + dist_to_cam
        obj.location.x = 0
        obj.location.z = 0
        obj.rotation_euler = (0, 0, 0)
        obj.scale[0] = w
        obj.scale[2] = h
        obj.scale[1] = 0

    def toggle_rain(self, is_clean):
        """完全按照指定路径修改参数，不对其他参数做任何多余操作"""
        if is_clean:
            # 物理隐藏玻璃，生成绝对干净的 GT
            self.glass_obj.hide_render = True
        else:
            # 显示玻璃，生成 Blended
            self.glass_obj.hide_render = False

            # --- 精确控制 GreenEges_Glass 材质 ---
            mat_green = bpy.data.materials.get("GreenEges_Glass")
            if mat_green and "Principled BSDF" in mat_green.node_tree.nodes:
                bsdf_green = mat_green.node_tree.nodes["Principled BSDF"]
                bsdf_green.inputs[4].default_value = 0.0
                bsdf_green.inputs[3].default_value = 1.0

            # --- 精确控制 Material.005 材质 ---
            mat_005 = bpy.data.materials.get("Material.005")
            if mat_005:
                nodes_005 = mat_005.node_tree.nodes

                # 设置 Noise Texture.001
                if "Noise Texture.001" in nodes_005:
                    nodes_005["Noise Texture.001"].inputs[2].default_value = random.uniform(0.0, 5.0)

                # 设置 Noise Texture
                if "Noise Texture" in nodes_005:
                    nodes_005["Noise Texture"].inputs[2].default_value = random.uniform(12.0, 22.0)

                # --- 新增：Mapping.002 ---
                if "Mapping.002" in nodes_005:
                    map_002 = nodes_005["Mapping.002"]
                    map_002.inputs[1].default_value[0] = random.uniform(0.0, 100.0)
                    map_002.inputs[1].default_value[1] = random.uniform(0.0, 100.0)
                    map_002.inputs[1].default_value[2] = random.uniform(0.0, 100.0)

                # --- 新增：Voronoi Texture ---
                if "Voronoi Texture" in nodes_005:
                    nodes_005["Voronoi Texture"].inputs[2].default_value = random.uniform(3.0, 5.0)

                # --- 新增：Mapping ---
                if "Mapping" in nodes_005:
                    map_base = nodes_005["Mapping"]
                    map_base.inputs[1].default_value[0] = random.uniform(0.0, 100.0)
                    map_base.inputs[1].default_value[1] = random.uniform(0.0, 100.0)
                    map_base.inputs[1].default_value[2] = random.uniform(0.0, 100.0)

    def run_batch(self, num_samples):
        img_dir = os.path.join(self.base_dir, "train2017", "*.jpg")
        all_images = glob(img_dir)
        self.get_scene_objects()

        start_time = time.time()
        for i in range(num_samples):
            bg_path = random.choice(all_images)
            img = bpy.data.images.load(bg_path)

            self.scene.render.resolution_x, self.scene.render.resolution_y = img.size
            tex_node = self.bg_plane.active_material.node_tree.nodes.get("ImageNode")
            if tex_node: tex_node.image = img

            bg_dist = 5.0
            self.setup_background(bg_dist)
            self.setup_glass(0.7, 3.0)

            sample_name = f"sample_{i:05d}"

            # --- [渲染 Clean] ---
            self.toggle_rain(is_clean=True)
            self.cam.data.dof.use_dof = False
            self.scene.render.filepath = os.path.join(self.path_clean, sample_name)
            bpy.ops.render.render(write_still=True)

            # --- [渲染 Blended] ---
            self.toggle_rain(is_clean=False)
            self.cam.data.dof.use_dof = True
            self.cam.data.dof.focus_distance = bg_dist
            self.cam.data.dof.aperture_fstop = random.uniform(22.0, 64.0)

            self.scene.render.filepath = os.path.join(self.path_blended, sample_name)
            bpy.ops.render.render(write_still=True)

            bpy.data.images.remove(img)
            print(f"✅ 已完成: {sample_name} | 耗时: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    MY_PROJECT_DIR = r"D:\AAA_Projects\text-based-denoising\image_synthesizer"
    gen = RainPBRGenerator(MY_PROJECT_DIR)
    gen.run_batch(num_samples=1000)