import os
import time
import argparse

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from models.DehazeXL.decoders.decoder import DehazeXL


def unnormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, total_params / 1e6, trainable_params, trainable_params / 1e6


def count_flops_thop(model, input_tensor):
    """
    Count FLOPs using thop.
    Returns:
        flops (float): FLOPs
        params (float): Params (thop's count, may differ slightly)
    """
    try:
        from thop import profile
    except ImportError as e:
        raise ImportError("thop is not installed. Run: pip install thop") from e

    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops, params


def human_readable_flops(flops):
    # flops in raw count (multiply-add counted as 1 op by thop; conventions vary)
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    # Build model (LGD / Dehaze module)
    model = DehazeXL().to(device)

    # Prepare model checkpoint list
    if os.path.isfile(args.model_path):
        model_path_list = [args.model_path]
    else:
        model_path_list = [os.path.join(args.model_path, p) for p in os.listdir(args.model_path)]

    # Prepare test images
    cloud_img_list = []
    folder_flag = 0
    if not os.path.exists(args.test_img):
        raise FileExistsError("Please input path of test img.")
    if os.path.isfile(args.test_img):
        cloud_img_list.append(args.test_img)
    else:
        folder_flag = 1
        cloud_img_list = os.listdir(args.test_img)

    # Detector info (given by you)
    det_params_m = args.det_params_m       # YOLO11s params
    det_gflops = args.det_gflops           # YOLO11s GFLOPs (you can fill if known, else 0)

    # Build transform
    transforms_list = [transforms.ToTensor()]
    if args.nor:
        transforms_list.append(
            transforms.Normalize(
                [0.48694769, 0.53490934, 0.45040282],
                [0.16451647, 0.15308164, 0.15011494]
            )
        )
    transform = transforms.Compose(transforms_list)

    for i, ckpt in tqdm(list(enumerate(model_path_list)), total=len(model_path_list)):
        ckpt = ckpt if os.path.exists(ckpt) else os.path.join(args.model_path, ckpt)
        print(f"\n{i+1}/{len(model_path_list)} - Loading model: {ckpt}")

        # Load weights
        if device.type == "cpu":
            state = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)
        else:
            state = torch.load(ckpt, weights_only=True)
        model.load_state_dict(state)

        # Count parameters
        total_p, total_m, train_p, train_m = count_parameters(model)
        print(f"[LGD Params] total: {total_p:,} ({total_m:.2f} M), trainable: {train_p:,} ({train_m:.2f} M)")

        # Prepare a dummy input for FLOPs counting
        # NOTE: FLOPs depends on input size; use args.flops_size (recommended <= 2048)
        flops_h = args.flops_h
        flops_w = args.flops_w

        dummy = torch.randn(1, 3, flops_h, flops_w, device=device)
        if args.fp16:
            model.half()
            dummy = dummy.half()

        # Count FLOPs (LGD)
        try:
            flops, thop_params = count_flops_thop(model, dummy)
            # Convert to GFLOPs (1e9)
            lgd_gflops = flops / 1e9
            print(f"[LGD FLOPs] input=({flops_h}x{flops_w}) => {human_readable_flops(flops)} ({lgd_gflops:.2f} GFLOPs)")
        except Exception as e:
            lgd_gflops = None
            print(f"[LGD FLOPs] Failed to compute FLOPs: {repr(e)}")

        # Two-stage totals (LGD + Detector)
        if det_params_m > 0:
            hae_total_m = total_m + det_params_m
            print(f"[HAE-Net Params] LGD({total_m:.2f}M) + Detector({det_params_m:.2f}M) = {hae_total_m:.2f} M")

            if lgd_gflops is not None and det_gflops > 0:
                hae_total_gflops = lgd_gflops + det_gflops
                print(f"[HAE-Net FLOPs] LGD({lgd_gflops:.2f}) + Detector({det_gflops:.2f}) = {hae_total_gflops:.2f} GFLOPs "
                      f" @ input=({flops_h}x{flops_w})")
            elif lgd_gflops is not None and det_gflops == 0:
                print(f"[HAE-Net FLOPs] LGD({lgd_gflops:.2f} GFLOPs) + Detector(unknown) -> please fill --det_gflops")
            else:
                print("[HAE-Net FLOPs] Unknown (LGD FLOPs unavailable)")

        # ---- Below is your original inference loop (optional) ----
        model.eval()

        start_time = time.time()
        for path in cloud_img_list:
            if folder_flag == 0:
                img_path = path
            else:
                img_path = os.path.join(args.test_img, path)

            os.makedirs(os.path.join(args.save_dir, f'{os.path.basename(img_path)}'), exist_ok=True)
            save_dir = os.path.join(args.save_dir, f'{os.path.basename(img_path)}')
            print(save_dir)

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
            out_name = f'{os.path.basename(ckpt)}_{os.path.basename(img_path)}'
            res.save(os.path.join(save_dir, out_name))

        end_time = time.time()
        avg_time = (end_time - start_time) / max(len(cloud_img_list), 1)
        print(f"Done. Avg Time: {avg_time:.6f}s per image")


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Test a model, count params and FLOPs.')
    parser.add_argument('--test_img', type=str,
                        default=r'./datasets/8KDehaze_mini/cloud_L1/16s14e09_c50y15_0.png',
                        help='Path to dataset (file or folder)')
    parser.add_argument('--model_path', type=str,
                        default=r'./checkpoints/conv_best1.pth',
                        help='Path to pretrained model (file or folder)')
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

    # Detector stats (you already have params; FLOPs can be filled later)
    parser.add_argument('--det_params_m', type=float, default=9.4,
                        help='Detector params in Million (M). e.g., YOLO11s=9.4')
    parser.add_argument('--det_gflops', type=float, default=0.0,
                        help='Detector GFLOPs at the same input size. Set if known.')

    # FLOPs input size (IMPORTANT!)
    parser.add_argument('--flops_h', type=int, default=2048,
                        help='Input height used for FLOPs counting (recommended <= 2048)')
    parser.add_argument('--flops_w', type=int, default=2048,
                        help='Input width used for FLOPs counting (recommended <= 2048)')

    main(parser.parse_args())
