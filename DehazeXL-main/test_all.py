import os
import time
import argparse

from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

from models.DehazeXL.decoders.decoder import DehazeXL


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def unnormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def collect_images(root: str, recursive: bool):
    """Collect image paths from a folder. Returns absolute paths."""
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMG_EXTS:
                files.append(p)
    files.sort()
    return files


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    model = DehazeXL().to(device)

    # model_path: allow a single file or a folder containing multiple weights
    if os.path.isfile(args.model_path):
        model_path_list = [args.model_path]
    else:
        model_path_list = [os.path.join(args.model_path, x) for x in os.listdir(args.model_path)]

    # test_img: allow a single file or a folder (batch)
    if not os.path.exists(args.test_img):
        raise FileExistsError("Please input path of test img.")

    if os.path.isfile(args.test_img):
        cloud_img_list = [args.test_img]
        input_root = os.path.dirname(args.test_img)
    else:
        input_root = args.test_img
        cloud_img_list = collect_images(args.test_img, recursive=args.recursive)

    if len(cloud_img_list) == 0:
        raise FileExistsError(f"No images found in: {args.test_img}")

    # build transform (same logic as your original code)
    transforms_list = [transforms.ToTensor()]
    if args.nor:
        transforms_list.append(
            transforms.Normalize(
                [0.48694769, 0.53490934, 0.45040282],
                [0.16451647, 0.15308164, 0.15011494]
            )
        )
    transform = transforms.Compose(transforms_list)

    for i, m in tqdm(list(enumerate(model_path_list)), desc="Models"):
        m = m if os.path.exists(m) else os.path.join(args.model_path, m)
        print("{}/{} - Loading model: {}".format(i + 1, len(model_path_list), m))

        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(m, map_location=torch.device('cpu'), weights_only=True))
        else:
            model.load_state_dict(torch.load(m, weights_only=True))

        if args.fp16:
            model.half()
            print("FP16 mode has been turned on")

        model.eval()

        start_time = time.time()

        # ✅ 去掉“模型名”目录前缀：直接写到 args.save_dir 下
        base_out_root = args.save_dir
        os.makedirs(base_out_root, exist_ok=True)

        for img_path in tqdm(cloud_img_list, desc="Images", leave=False):
            # Keep subfolder structure (if test_img is a folder)
            if os.path.isdir(args.test_img):
                rel = os.path.relpath(img_path, input_root)
                rel_dir = os.path.dirname(rel)
                save_dir = os.path.join(base_out_root, rel_dir)
            else:
                save_dir = base_out_root

            os.makedirs(save_dir, exist_ok=True)

            cloud_img = Image.open(img_path).convert('RGB')

            if args.uni_size_w > 0 and args.uni_size_h > 0:
                cloud_img = cloud_img.resize((args.uni_size_w, args.uni_size_h))

            print("Cloud removing...")
            with torch.no_grad():
                input_tensor = transform(cloud_img).unsqueeze(0).to(device)
                if args.fp16:
                    input_tensor = input_tensor.half()

                out = model(input_tensor)

                if args.nor:
                    out = unnormalize(
                        out,
                        [0.48694769, 0.53490934, 0.45040282],
                        [0.16451647, 0.15308164, 0.15011494]
                    )

                res = transforms.ToPILImage()(out.clamp(0, 1).squeeze(0))

            print("Saving result...")
            # ✅ Output name equals input name (unchanged)
            out_name = os.path.basename(img_path)
            res.save(os.path.join(save_dir, out_name))

        end_time = time.time()
        avg_time = (end_time - start_time) / len(cloud_img_list)
        print("Done. Avg Time:{}".format(avg_time))


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Test a model (batch folder supported).')
    parser.add_argument('--test_img', type=str,
                        default=r'./datasets/8KDehaze_mini/cloud_L1/16s14e09_c50y15_0.png',
                        help='Path to test image OR a folder containing images')
    parser.add_argument('--model_path', type=str,
                        default=r'./checkpoints/conv_best1.pth',
                        help='Path to pretrained model (.pth) OR a folder containing .pth files')
    parser.add_argument('--save_dir', type=str, default=r'./res',
                        help='Path to save predict results')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--uni_size_w', type=int, default=4096,
                        help='Unified input image size w')
    parser.add_argument('--uni_size_h', type=int, default=4096,
                        help='Unified input image size h')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize input image')
    parser.add_argument('--fp16', action='store_true',
                        help='Loading models and data in FP16 format')

    # Batch folder scanning
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively scan test_img folder for images')

    main(parser.parse_args())
