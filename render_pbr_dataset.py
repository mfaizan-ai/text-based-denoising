"""
render_pbr_dataset.py
---------------------
Physically Based Rendering (PBR) pipeline for reflection removal training data.
Implements the method described in the WindowSeat paper using Blender's
Principled BSDF shader to simulate realistic glass reflections.

Run from the command line using Blender's Python interpreter:

    blender --background --python render_pbr_dataset.py -- \\
        --transmission-dir data/raw/natural_images/ \\
        --reflection-dir   data/raw/natural_images/ \\
        --hdr-dir          data/raw/hdri/ \\
        --output-dir       data/train/pbr/ \\
        --n-samples        8000

Or install bpy (headless Blender) and run as a normal Python script:

    pip install bpy
    python render_pbr_dataset.py --transmission-dir ... --n-samples 8000

What this script simulates
---------------------------
For each training sample:
  1. Load a transmission image as a flat textured mesh plane (the scene)
  2. Load a reflection source — either an HDR panoramic map or a flat RGB image
  3. Set up a glass material using Principled BSDF with randomised parameters:
       - Index of Refraction (IoR): controls reflection strength
       - Roughness: controls blur of specular highlights
       - Glass thickness: controls ghosting from multiple internal reflections
  4. Render WITH glass -> blended image B
  5. Render WITHOUT glass (IoR=1.0, metallic=0, roughness=0) -> clean image T
  6. Save (B, T) pair

Glass parameters sampled per image:
  IoR       ~ U[1.45, 1.65]   (typical window glass: ~1.52)
  roughness ~ U[0.00, 0.20]   (0=mirror-sharp, 0.2=slightly frosted)
  thickness ~ U[0.002, 0.02]  (controls ghosting — thicker = more ghosting)
  metallic  = 0               (glass is dielectric, not metallic)

Output structure:
  output-dir/
    blended/   <- reflected images (ground truth input to model)
    clean/     <- clean transmission images (ground truth output)
"""

import argparse
import math
import os
import random
import sys

import numpy as np

# ── Import bpy (Blender Python API)  ─────────────────────────────────────────
try:
    import bpy
    import mathutils
    HAS_BPY = True
except ImportError:
    HAS_BPY = False
    print("WARNING: bpy not found.")
    print("Install with: pip install bpy")
    print("Or run via: blender --background --python render_pbr_dataset.py -- [args]")


# ── Rendering configuration  ──────────────────────────────────────────────────

RENDER_WIDTH  = 512     # Output image width  (increase for higher quality)
RENDER_HEIGHT = 512     # Output image height
SAMPLES       = 64      # Cycles render samples (higher = cleaner, slower)
                        # 64 is a good balance for training data generation


# ── Args  ─────────────────────────────────────────────────────────────────────

def get_args():
    # When called via blender --python script.py -- [args],
    # Blender consumes everything before --, so we parse after it.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        description="PBR glass reflection dataset generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transmission-dir", required=True,
                   help="Directory of transmission (background scene) images")
    p.add_argument("--reflection-dir",   default=None,
                   help="Directory of flat RGB reflection source images")
    p.add_argument("--hdr-dir",          default=None,
                   help="Directory of HDR panoramic .hdr or .exr files")
    p.add_argument("--output-dir",       required=True,
                   help="Output directory for (blended, clean) pairs")
    p.add_argument("--n-samples",        type=int, default=4000,
                   help="Number of pairs to generate")
    p.add_argument("--render-width",     type=int, default=RENDER_WIDTH)
    p.add_argument("--render-height",    type=int, default=RENDER_HEIGHT)
    p.add_argument("--samples",          type=int, default=SAMPLES,
                   help="Cycles render samples per image")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--start-idx",        type=int, default=0,
                   help="Start index (for resuming interrupted generation)")
    return p.parse_args(argv)


# ── Blender scene setup  ──────────────────────────────────────────────────────

def reset_scene():
    """Clear the default Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        bpy.data.images.remove(block)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block)


def setup_renderer(width: int, height: int, samples: int):
    """Configure Cycles renderer."""
    scene = bpy.context.scene
    scene.render.engine             = "CYCLES"
    scene.cycles.samples            = samples
    scene.cycles.use_denoising      = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_x       = width
    scene.render.resolution_y       = height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform  = "Standard"
    scene.view_settings.look            = "None"

    # Use GPU if available
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.refresh_devices()
    for device in prefs.devices:
        device.use = True
    scene.cycles.device = "GPU"


def add_camera():
    """
    Add an orthographic camera facing the glass plane head-on.
    Orthographic avoids perspective distortion which would misalign
    the blended and clean renders.
    """
    bpy.ops.object.camera_add(location=(0, -2, 0))
    cam = bpy.context.active_object
    cam.data.type  = "ORTHO"
    cam.data.ortho_scale = 2.0
    cam.rotation_euler   = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam
    return cam


def create_image_plane(image_path: str, name: str, z_offset: float = 0.0):
    """
    Creates a flat mesh plane textured with the given image.
    Used for both the transmission scene plane and the reflection source plane.
    """
    img  = bpy.data.images.load(image_path)
    W, H = img.size
    aspect = W / H

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, z_offset, 0))
    plane = bpy.context.active_object
    plane.name  = name
    plane.scale = (aspect, 1.0, 1.0)

    # Create emission material so the plane emits its texture as light
    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node   = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    emit_node  = nodes.new("ShaderNodeEmission")
    emit_node.inputs["Strength"].default_value = 1.0
    out_node   = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex_node.outputs["Color"],   emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    plane.data.materials.append(mat)
    return plane


def create_glass_plane(ior: float, roughness: float, thickness: float):
    """
    Creates the glass material plane positioned between camera and scene.

    Uses Principled BSDF with:
      - Transmission = 1.0    (fully transmissive base)
      - IOR                   (Index of Refraction — controls reflection strength)
      - Roughness              (microfacet blur of specular highlights)
      - thickness via geometry (controls ghosting from internal reflections)

    The glass is positioned at z=0, between the camera (z=-2) and
    the transmission plane (z=0.5).
    """
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, -0.05, 0))
    glass_plane = bpy.context.active_object
    glass_plane.name = "glass"
    glass_plane.rotation_euler = (math.radians(90), 0, 0)

    # Extrude slightly to give the glass physical thickness for ghosting
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, thickness, 0)}
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new(name="glass_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out  = nodes.new("ShaderNodeOutputMaterial")

    # Glass parameters
    bsdf.inputs["IOR"].default_value            = ior
    bsdf.inputs["Roughness"].default_value      = roughness
    bsdf.inputs["Metallic"].default_value       = 0.0
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Alpha"].default_value          = 1.0

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mat.blend_method    = "BLEND"
    mat.shadow_method   = "NONE"
    glass_plane.data.materials.append(mat)

    return glass_plane


def set_hdr_world(hdr_path: str, rotation_z: float = 0.0):
    """Set an HDR panoramic image as the world environment light."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg   = nodes.new("ShaderNodeBackground")
    env  = nodes.new("ShaderNodeTexEnvironment")
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    out  = nodes.new("ShaderNodeOutputWorld")

    env.image = bpy.data.images.load(hdr_path)
    mapping.inputs["Rotation"].default_value[2] = rotation_z

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"],  env.inputs["Vector"])
    links.new(env.outputs["Color"],       bg.inputs["Color"])
    links.new(bg.outputs["Background"],   out.inputs["Surface"])

    bg.inputs["Strength"].default_value = 1.0


def render_to_path(output_path: str):
    """Render the current scene to a file."""
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


# ── Per-sample rendering  ─────────────────────────────────────────────────────

def render_pair(
    trans_path  : str,
    refl_path   : str | None,
    hdr_path    : str | None,
    out_blended : str,
    out_clean   : str,
    width       : int,
    height      : int,
    samples     : int,
):
    """
    Render one (blended, clean) pair.

    Scene setup:
      Camera (orthographic, z=-2, facing +y)
        -> Glass plane (z=-0.05, with Principled BSDF)
           -> Transmission scene plane (z=0.5, emissive texture)
      World environment: HDR or flat RGB reflection source
    """
    reset_scene()
    setup_renderer(width, height, samples)
    add_camera()

    # Randomise glass parameters
    ior       = random.uniform(1.45, 1.65)
    roughness = random.uniform(0.00, 0.20)
    thickness = random.uniform(0.002, 0.02)

    # Set world lighting — HDR takes priority over flat reflection image
    if hdr_path is not None:
        rot_z = random.uniform(0, 2 * math.pi)
        set_hdr_world(hdr_path, rotation_z=rot_z)
    elif refl_path is not None:
        # Use flat RGB image as a planar reflection source behind the glass
        # Place it far back so it acts as environment
        create_image_plane(refl_path, "reflection_source", z_offset=2.0)
    else:
        # Fallback: neutral grey world
        bpy.context.scene.world.color = (0.5, 0.5, 0.5)

    # Transmission plane
    create_image_plane(trans_path, "transmission", z_offset=0.5)

    # ── Render WITH glass ─────────────────────────────────────────────────────
    glass = create_glass_plane(ior, roughness, thickness)
    render_to_path(out_blended)

    # ── Render WITHOUT glass (clean ground truth) ─────────────────────────────
    # Remove glass and set IoR=1.0 equivalent by deleting the glass object
    bpy.data.objects.remove(glass, do_unlink=True)
    render_to_path(out_clean)


# ── Main loop  ────────────────────────────────────────────────────────────────

def collect_images(directory: str | None, exts=(".jpg", ".jpeg", ".png")) -> list:
    if directory is None or not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    ])


def collect_hdrs(directory: str | None) -> list:
    if directory is None or not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".hdr", ".exr"))
    ])


def main():
    if not HAS_BPY:
        print("\nCannot render without bpy. Install with:")
        print("  pip install bpy")
        print("Or run as:")
        print("  blender --background --python render_pbr_dataset.py -- [args]")
        sys.exit(1)

    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    blended_dir = os.path.join(args.output_dir, "blended")
    clean_dir   = os.path.join(args.output_dir, "clean")
    os.makedirs(blended_dir, exist_ok=True)
    os.makedirs(clean_dir,   exist_ok=True)

    trans_imgs  = collect_images(args.transmission_dir)
    refl_imgs   = collect_images(args.reflection_dir)
    hdr_imgs    = collect_hdrs(args.hdr_dir)

    print(f"Transmission images : {len(trans_imgs)}")
    print(f"Reflection images   : {len(refl_imgs)}")
    print(f"HDR maps            : {len(hdr_imgs)}")
    print(f"Output dir          : {args.output_dir}")
    print(f"Samples to generate : {args.n_samples}")
    print(f"Render resolution   : {args.render_width}x{args.render_height}")
    print(f"Cycles samples      : {args.samples}")

    if not trans_imgs:
        print("ERROR: No transmission images found.")
        sys.exit(1)

    if not refl_imgs and not hdr_imgs:
        print("WARNING: No reflection sources (RGB or HDR). "
              "Glass will reflect the world background only.")

    n_existing = len([f for f in os.listdir(blended_dir) if f.endswith(".png")])
    print(f"Already generated   : {n_existing}")

    for i in range(args.start_idx + n_existing, args.n_samples):
        stem        = f"pbr_{i:07d}"
        out_blended = os.path.join(blended_dir, f"{stem}.png")
        out_clean   = os.path.join(clean_dir,   f"{stem}.png")

        if os.path.exists(out_blended) and os.path.exists(out_clean):
            continue

        trans_path = random.choice(trans_imgs)

        # Choose reflection source: HDR (70%) or flat RGB (30%)
        if hdr_imgs and (not refl_imgs or random.random() < 0.7):
            hdr_path  = random.choice(hdr_imgs)
            refl_path = None
        elif refl_imgs:
            hdr_path  = None
            refl_path = random.choice(refl_imgs)
            # Reflection must be different from transmission
            while refl_path == trans_path and len(refl_imgs) > 1:
                refl_path = random.choice(refl_imgs)
        else:
            hdr_path  = None
            refl_path = None

        print(f"  [{i+1}/{args.n_samples}]  {stem}  "
              f"refl={'HDR' if hdr_path else 'RGB' if refl_path else 'none'}")

        try:
            render_pair(
                trans_path=trans_path,
                refl_path=refl_path,
                hdr_path=hdr_path,
                out_blended=out_blended,
                out_clean=out_clean,
                width=args.render_width,
                height=args.render_height,
                samples=args.samples,
            )
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            continue

    print(f"\nDone. Generated pairs in: {args.output_dir}")
    print("Add to training data:")
    print("  train_windowseat.py --data-root data/ will pick these up")
    print("  if output-dir is data/train/pbr/ or data/val/pbr/")


if __name__ == "__main__":
    main()