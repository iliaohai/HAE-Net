import os
import numpy as np
import cv2
from tqdm import tqdm

# --------- Haze levels (1-6): you can tune these presets ----------
# Larger beta  -> heavier haze (shorter visibility)
# Larger brightness -> brighter/whiter airlight
HAZE_PRESETS = {
    1: {"beta": 0.03, "brightness": 0.55},  # very light
    2: {"beta": 0.05, "brightness": 0.60},  # light
    3: {"beta": 0.07, "brightness": 0.70},  # medium
    4: {"beta": 0.09, "brightness": 0.80},  # medium-heavy
    5: {"beta": 0.11, "brightness": 0.90},  # heavy
    6: {"beta": 0.13, "brightness": 0.95},  # very heavy
}


def add_hazy(image_bgr: np.ndarray, beta: float, brightness: float) -> np.ndarray:
    """
    Add haze to a single image using a simple atmospheric scattering approximation.

    Args:
        image_bgr: input image in BGR, uint8
        beta: haze density (higher -> heavier haze)
        brightness: airlight intensity in [0, 1] (higher -> brighter/whiter haze)

    Returns:
        hazy image in BGR, uint8
    """
    img_f = image_bgr.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]

    # Make transmission vary with distance to image center (similar to your original code)
    size = np.sqrt(max(h, w))
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    d = -0.04 * dist + size
    t = np.exp(-beta * d)  # transmission map in (0,1]
    t = np.clip(t, 0.0, 1.0)

    # Apply airlight model: I_hazy = I * t + A * (1 - t)
    img_hazy = img_f * t[..., None] + brightness * (1.0 - t[..., None])
    img_hazy = np.clip(img_hazy * 255.0, 0, 255).astype(np.uint8)
    return img_hazy


def process_folder_six_levels(input_folder: str, output_root: str, keep_structure: bool = True):
    """
    For each input image, generate 6 hazy versions (level 1..6) and save to:
      output_root/level_1/xxx.png
      ...
      output_root/level_6/xxx.png

    Args:
        input_folder: folder containing images
        output_root: root folder for outputs
        keep_structure: if True, keep subfolder structure under input_folder
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    # Collect all image paths (recursive)
    image_paths = []
    for dirpath, _, filenames in os.walk(input_folder):
        for fn in filenames:
            if fn.lower().endswith(valid_ext):
                image_paths.append(os.path.join(dirpath, fn))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {input_folder}")

    # Prepare output folders
    for level in range(1, 7):
        os.makedirs(os.path.join(output_root, f"level_{level}"), exist_ok=True)

    for img_path in tqdm(image_paths, desc="Generating hazy images (levels 1-6)"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: failed to read {img_path}, skipped.")
            continue

        # Compute relative path (to keep directory structure if needed)
        rel_path = os.path.relpath(img_path, input_folder) if keep_structure else os.path.basename(img_path)
        rel_dir = os.path.dirname(rel_path)

        for level in range(1, 7):
            beta = HAZE_PRESETS[level]["beta"]
            brightness = HAZE_PRESETS[level]["brightness"]

            hazy = add_hazy(img, beta=beta, brightness=brightness)

            out_dir = os.path.join(output_root, f"level_{level}", rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(rel_path))

            cv2.imwrite(out_path, hazy)

    print(f"\nDone. Saved 6-level hazy images to: {output_root}")
    print("Folders:")
    for level in range(1, 7):
        print(f"  - {os.path.join(output_root, f'level_{level}')}  (beta={HAZE_PRESETS[level]['beta']}, "
              f"brightness={HAZE_PRESETS[level]['brightness']})")


if __name__ == "__main__":
    # Input folder (original images)
    input_dir = "input"

    # Output root: will create level_1 ... level_6
    output_root = "output"

    process_folder_six_levels(input_dir, output_root, keep_structure=True)
