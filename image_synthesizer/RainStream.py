import bpy
import os
import random
import time
import math
from glob import glob
from mathutils import Vector


class RainStreamGenerator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.scene = bpy.context.scene
        self.setup_render_engine()

        self.path_blended = os.path.join(base_dir, "rainstream", "syn_zka", "blended")
        self.path_clean = os.path.join(base_dir, "rainstream", "syn_zka", "clean")
        for p in [self.path_blended, self.path_clean]:
            os.makedirs(p, exist_ok=True)

    def setup_render_engine(self):
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = 128
        self.scene.cycles.use_denoising = True

    def get_scene_objects(self):
        self.bg_plane = bpy.data.objects.get("Background")
        self.glass_obj = bpy.data.objects.get("RainPlain")  # 改为你的物体名
        self.cam = bpy.data.objects.get("Camera")

        # 增加安全性检查，防止找不到物体报错
        if not self.glass_obj:
            print("❌ 错误: 找不到 RainPlain 物体")
            return

        self.glass_nodes = self.glass_obj.active_material.node_tree.nodes

        # 对应你之前手动建立的雨线节点名
        self.mapping_node = self.glass_nodes.get("映射") or self.glass_nodes.get("Mapping")
        self.color_ramp_node = self.glass_nodes.get("颜色渐变") or self.glass_nodes.get("ColorRamp")

    def calculate_view_size(self, distance):
        """完全保留原版物理缩放逻辑"""
        cam_data = self.cam.data
        cam_data.sensor_fit = 'VERTICAL'
        aspect_ratio = self.scene.render.resolution_x / self.scene.render.resolution_y

        v_fov_half = math.atan(cam_data.sensor_height / (2 * cam_data.lens))
        half_height = math.tan(v_fov_half) * distance
        half_width = half_height * aspect_ratio

        # 保留原有的 1.15 倍冗余
        return half_width * 1.15, half_height * 1.15

    def setup_background(self, distance):
        """完全保留原版背景布局逻辑"""
        w, h = self.calculate_view_size(distance)
        obj = self.bg_plane
        obj.location = (0, self.cam.location.y + distance, 0)
        obj.rotation_euler = (math.radians(90), 0, 0)
        obj.scale[0] = w
        obj.scale[1] = h
        obj.scale[2] = 1.0

    def setup_glass(self, min_dist=1.0, max_dist=1.1):
        """修正旋转逻辑：确保平面正对相机"""
        dist_to_cam = random.uniform(min_dist, max_dist)
        w, h = self.calculate_view_size(dist_to_cam)
        obj = self.glass_obj

        # 位置保持原样
        obj.location.y = self.cam.location.y + dist_to_cam
        obj.location.x = 0
        obj.location.z = 0

        obj.rotation_euler = (math.radians(90), math.radians(90), 0)

        obj.scale[0] = w
        obj.scale[2] = h
        obj.scale[1] = 1.0  # 恢复一点厚度或保持 1.0 避免缩放塌陷

    def toggle_rain(self, is_clean):
        """
        基于用户精确参数优化的2D雨线逻辑
        """
        if is_clean:
            # 物理隐藏平面，确保 GT 100% 干净
            self.glass_obj.hide_render = True
        else:
            # 渲染雨线
            self.glass_obj.hide_render = False

            # --- 1. 控制物体旋转 (RainPlain 绕 Y 轴旋转) ---
            # 基准 90度 (math.pi/2)，偏移 +-30度 (math.pi/6)
            base_rot = math.radians(90)
            offset_rot = math.radians(random.uniform(-30.0, 30.0))
            self.glass_obj.rotation_euler[1] = base_rot + offset_rot

            # --- 2. 控制映射节点 (Mapping) 的缩放参数 ---
            if self.mapping_node:
                # inputs[3] 是缩放 (Scale)
                # X轴: -2.0 到 0.2
                self.mapping_node.inputs[3].default_value[0] = random.uniform(2.0, 25.0)
                # Y轴: 40.0 到 70.0
                self.mapping_node.inputs[3].default_value[1] = random.uniform(100.0, 400.0)
                # Z轴: 0.0 到 100.0 (随机数值)
                self.mapping_node.inputs[3].default_value[2] = random.uniform(0.0, 100.0)

                # 同时建议随机化位置 (Location) 以增加每一帧的变化
                self.mapping_node.inputs[1].default_value[0] = random.uniform(0, 100)
                self.mapping_node.inputs[1].default_value[1] = random.uniform(0, 100)

            # --- 3. 控制颜色渐变 (ColorRamp) 的滑块位置 ---
            if self.color_ramp_node:
                # 黑色滑块 (低值): 0.55 - 0.75
                low_val = random.uniform(0.55, 0.75)
                self.color_ramp_node.color_ramp.elements[0].position = low_val

                # 白色滑块 (高值): 0.85 - 1.00
                high_val = random.uniform(0.85, 1.00)
                self.color_ramp_node.color_ramp.elements[1].position = high_val

    def run_batch(self, num_samples):
        """完全保留原版循环和先 GT 后 Blended 的逻辑"""
        img_dir = os.path.join(self.base_dir, "train2017", "*.jpg")
        all_images = glob(img_dir)
        self.get_scene_objects()

        start_time = time.time()
        for i in range(num_samples):
            bg_path = random.choice(all_images)
            print(bg_path)
            img = bpy.data.images.load(bg_path)

            # 1. 设置分辨率与纹理
            self.scene.render.resolution_x, self.scene.render.resolution_y = img.size
            tex_node = self.bg_plane.active_material.node_tree.nodes.get("ImageNode")
            tex_node.image = img

            # 2. 物理布局 (背景固定5m，雨平面 1.0-1.1m)
            bg_dist = 5.0
            self.setup_background(bg_dist)
            self.setup_glass(4.0, 4.1)

            sample_name = f"sample_{i:05d}"

            # --- [渲染 Clean (GT)] ---
            self.toggle_rain(is_clean=True)
            self.cam.data.dof.use_dof = False  # 彻底关闭景深
            self.scene.render.filepath = os.path.join(self.path_clean, sample_name)
            bpy.ops.render.render(write_still=True)

            # --- [渲染 Blended (Rain)] ---
            self.toggle_rain(is_clean=False)
            self.cam.data.dof.use_dof = True
            self.cam.data.dof.focus_distance = bg_dist  # 对焦在背景
            self.cam.data.dof.aperture_fstop = random.uniform(22.0, 64.0)

            self.scene.render.filepath = os.path.join(self.path_blended, sample_name)
            bpy.ops.render.render(write_still=True)

            # 3. 清理内存
            bpy.data.images.remove(img)
            print(f"✅ 已完成: {sample_name} | 耗时: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    MY_PROJECT_DIR = r"D:\AAA_Projects\text-based-denoising\image_synthesizer"
    gen = RainStreamGenerator(MY_PROJECT_DIR)
    gen.run_batch(num_samples=1000)