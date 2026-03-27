import bpy
import os
import random
import time
from glob import glob


class PBRDataGenerator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.scene = bpy.context.scene
        self.setup_render_engine()

        # Output directory setup
        self.path_blended = os.path.join(base_dir, "reflection", "syn", "blended")
        self.path_clean = os.path.join(base_dir, "reflection", "syn", "clean")

        for p in [self.path_blended, self.path_clean]:
            os.makedirs(p, exist_ok=True)

    def setup_render_engine(self):
        """Force GPU activation and optimize render settings"""
        self.scene.render.engine = 'CYCLES'

        # 1. Access Cycles preferences
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences

        # 2. Set compute device type (OPTIX > CUDA > HIP > METAL)
        for device_type in ['OPTIX', 'CUDA', 'HIP', 'METAL']:
            try:
                cycles_prefs.get_devices_for_type(device_type)
                cycles_prefs.compute_device_type = device_type
                print(f"✅ Compute platform activated: {device_type}")
                break
            except:
                continue

        # 3. Enable all available GPU devices
        for device in cycles_prefs.devices:
            if device.type in {'CUDA', 'OPTIX', 'HIP', 'METAL'}:
                device.use = True
                print(f"🖥️  Device enabled: {device.name}")
            else:
                device.use = False  # Disable CPU to prevent bottlenecking

        self.scene.cycles.device = 'GPU'

        # --- High-speed rendering optimizations ---
        self.scene.cycles.samples = 16
        self.scene.cycles.use_adaptive_sampling = True
        self.scene.cycles.max_bounces = 2
        self.scene.render.use_persistent_data = True

    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for mat in bpy.data.materials: bpy.data.materials.remove(mat)
        for img in bpy.data.images: bpy.data.images.remove(img)

    def create_plane(self, name, location, rotation, is_emissive=False):
        """Initialize plane with basic PBR or Emissive material"""
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=location, rotation=rotation)
        plane = bpy.context.active_object
        plane.name = name

        mat = bpy.data.materials.new(name=f"Mat_{name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        tex = nodes.new('ShaderNodeTexImage')
        tex.name = "ImageNode"

        emit = nodes.new('ShaderNodeEmission')
        emit.name = "EmissionNode"

        emit.inputs['Strength'].default_value = 1.0
        out = nodes.new('ShaderNodeOutputMaterial')

        mat.node_tree.links.new(tex.outputs['Color'], emit.inputs['Color'])
        mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
        plane.data.materials.append(mat)
        return plane

    def setup_camera(self, w, h):
        if not bpy.data.objects.get("Camera"):
            bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
        cam = bpy.context.active_object
        self.scene.camera = cam
        cam.data.type = 'ORTHO'
        cam.data.sensor_fit = 'VERTICAL'
        cam.data.ortho_scale = 1.0
        self.scene.render.resolution_x, self.scene.render.resolution_y = w, h

    def create_glass(self):
        """Create glass plane with Solidify modifier for Ghosting effect"""
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, -2, 0), rotation=(1.5708, 0, 0))
        glass = bpy.context.active_object
        glass.name = "GlassPlane"
        glass.scale *= 20.0

        # Solidify modifier handles 'Thickness' for Ghosting (Sec 3.2)
        glass.modifiers.new(name="Solidify", type='SOLIDIFY')

        mat = bpy.data.materials.new(name="Mat_Glass")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        out = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

        # Initial transmission settings
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = 1.0
        else:
            bsdf.inputs['Transmission'].default_value = 1.0

        glass.data.materials.append(mat)

    def update_plane_texture(self, plane, image_path, target_ratio=None):
        """Update image texture and adapt aspect ratio (Cover mode for reflection)"""
        img = bpy.data.images.load(image_path)
        nodes = plane.active_material.node_tree.nodes
        tex_node = nodes.get("ImageNode")
        tex_node.image = img

        w, h = img.size
        current_ratio = w / h

        if target_ratio is None:
            # Background logic: Keep original ratio
            plane.scale[0] = current_ratio
            plane.scale[1] = 1.0
        else:
            # Reflection source logic: Ensure background coverage (Cover)
            scale_factor = max(1.0, target_ratio / current_ratio)
            plane.scale[0] = current_ratio * scale_factor
            plane.scale[1] = 1.0 * scale_factor

            # Safety margin for random rotation to avoid visible edges
            plane.scale *= 1.25

        return w, h

    def run_batch(self, num_samples):
        # Scan for images
        img_dir = os.path.join(self.base_dir, "train2017", "*.jpg")
        all_images = glob(img_dir)
        if len(all_images) < 2:
            print("Error: Not enough images in the library.")
            return

        self.clear_scene()

        # 1. Initialize scene objects
        bg_plane = self.create_plane("Background", (0, 0, 0), (1.5708, 0, 0))
        refl_plane = self.create_plane("ReflectionSource", (0, -10, 0), (-1.5708, 0, 0), is_emissive=True)

        # 2. Initialize glass
        self.create_glass()
        glass_obj = bpy.data.objects["GlassPlane"]
        bsdf = next((n for n in glass_obj.active_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)

        start_time = time.time()
        print(f"\n🚀 Starting PBR Generation Task - Target: {num_samples} sets")
        print("-" * 70)

        for i in range(num_samples):
            # Sample pair
            sample_pair = random.sample(all_images, 2)
            bg_path, refl_path = sample_pair[0], sample_pair[1]
            sample_name = f"sync_{i:05d}"

            # --- STEP 1: Update textures and camera ---
            bg_w, bg_h = self.update_plane_texture(bg_plane, bg_path)
            bg_ratio = bg_w / bg_h
            self.setup_camera(bg_w, bg_h)
            self.update_plane_texture(refl_plane, refl_path, target_ratio=bg_ratio)

            # --- STEP 2: Randomize Reflection Angles ---
            rand_range = 0.25
            refl_plane.rotation_euler = (
                -1.5708 + random.uniform(-rand_range, rand_range),
                random.uniform(-rand_range, rand_range),
                random.uniform(-rand_range, rand_range)
            )

            # --- STEP 3: Randomize PBR Parameters (Sec 3.2) ---
            r_ior = random.uniform(1.25, 1.75)  # Paper range
            r_rough = random.uniform(0.0, 0.05)  # Paper range
            r_thick = random.uniform(0.0, 0.05)  # 0 to 5cm for Ghosting

            # Reflection intensity control
            r_metal = random.uniform(0.0, 0.1)
            # Light attenuation (Base Color)
            r_color = (random.uniform(0.9, 1.0), random.uniform(0.9, 1.0), random.uniform(0.9, 1.0), 1.0)
            # Emission strength (Radiance simulation)
            r_emit = random.uniform(0.5, 3.0) if random.random() > 0.2 else random.uniform(4.0, 8.0)

            # --- STEP 4: Render BLENDED (Reflection Contaminated) ---
            refl_plane.hide_render = False

            # Get the emission node safely by the name defined in create_plane
            refl_nodes = refl_plane.active_material.node_tree.nodes
            refl_emit = refl_nodes.get("EmissionNode")

            if refl_emit:
                refl_emit.inputs['Strength'].default_value = r_emit
            else:
                print(f"Warning: EmissionNode not found on {refl_plane.name}")

            # Apply PBR parameters from the paper
            glass_obj.modifiers["Solidify"].thickness = r_thick
            bsdf.inputs['IOR'].default_value = r_ior
            bsdf.inputs['Roughness'].default_value = r_rough

            # Handle Blender version differences for Metallic and Base Color
            if 'Metallic' in bsdf.inputs:
                bsdf.inputs['Metallic'].default_value = r_metal

            # Use 'Base Color' for light attenuation simulation (Sec 3.2)
            bsdf.inputs['Base Color'].default_value = r_color

            # Set path and render
            self.scene.render.filepath = os.path.join(self.path_blended, f"{sample_name}.png")
            bpy.ops.render.render(write_still=True)

            # --- STEP 5: Render CLEAN (Ground Truth) ---
            refl_plane.hide_render = True
            bsdf.inputs['IOR'].default_value = 1.0  # Invisible glass
            bsdf.inputs['Roughness'].default_value = 0.0
            if 'Metallic' in bsdf.inputs: bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)

            self.scene.render.filepath = os.path.join(self.path_clean, f"{sample_name}.png")
            bpy.ops.render.render(write_still=True)

            # --- STEP 6: Progress & Memory Management ---
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            print(f"Progress: {((i + 1) / num_samples) * 100:5.1f}% | Speed: {avg_time:.2f}s/set | ID: {sample_name}",
                  end='\r')

            # Clean image cache to prevent RAM overflow
            for node_name in ["Background", "ReflectionSource"]:
                obj = bpy.data.objects.get(node_name)
                img = obj.active_material.node_tree.nodes["ImageNode"].image
                if img: bpy.data.images.remove(img)

        print(f"\n\n✅ Task Completed! Total Time: {elapsed / 60:.2f} minutes")


# --- EXECUTION ---
if __name__ == "__main__":
    MY_PROJECT_DIR = r"D:\AAA_Projects\text-based-denoising\image_synthesizer"

    gen = PBRDataGenerator(MY_PROJECT_DIR)
    gen.run_batch(num_samples=10)