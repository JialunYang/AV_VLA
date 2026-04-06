import argparse
from pathlib import Path
import sys

import torch
from nuscenes import NuScenes
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENEMMA_ROOT = REPO_ROOT / "OpenEMMA"

for path in (SCRIPT_DIR, REPO_ROOT, OPENEMMA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from main_v3 import SceneSpecialistBranch
from scene_token_generator import generate_scene_tokens
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.6-mistral-7b"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output-file", type=str, default="tokens_20_samples.txt")
    return parser.parse_args()


def load_visual_model(model_path: str):
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava-v1.6-mistral-7b",
    )
    model.eval()
    return tokenizer, model, image_processor


def extract_image_embeds(image_path: str, image_processor, model) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)

    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]

    if image_tensor.ndim == 5 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims before encode_images, got {tuple(image_tensor.shape)}")

    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device

    dtype = getattr(model, "dtype", torch.float16)

    with torch.inference_mode():
        image_embeds = model.encode_images(image_tensor.to(device=device, dtype=dtype))

    image_embeds = image_embeds.squeeze(0)

    # LLaVA any-res inputs can return one image as (tiles, patches, dim).
    # The current SceneSpecialistBranch expects a flat token table (N, D),
    # so collapse all token-producing dimensions except the feature dim.
    if image_embeds.ndim > 2:
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

    if image_embeds.ndim != 2:
        raise ValueError(f"Expected image_embeds with shape (N, D), got {tuple(image_embeds.shape)}")

    return image_embeds


def build_output_block(sample_index: int, sample_token: str, tokens) -> str:
    lines = [f"Sample {sample_index}: {sample_token}"]
    lines.extend(tokens)
    return "\n".join(lines)


def main():
    args = parse_args()
    output_path = Path(args.output_file).resolve()

    print(f"Loading model from: {args.model_path}")
    _, model, image_processor = load_visual_model(args.model_path)

    print(f"Loading nuScenes {args.version} from: {args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    specialist_branch = None
    output_blocks = []

    samples = nusc.sample[: args.num_samples]
    for sample_index, sample in enumerate(samples, start=1):
        sample_token = sample["token"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        cam_front = nusc.get("sample_data", cam_front_token)
        image_path = str(Path(args.dataroot) / cam_front["filename"])

        print(f"[{sample_index}/{args.num_samples}] Processing sample {sample_token}")

        image_embeds = extract_image_embeds(image_path, image_processor, model).float()

        if specialist_branch is None:
            specialist_branch = SceneSpecialistBranch(
                input_dim=image_embeds.shape[-1],
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
            ).to(device=image_embeds.device)
            specialist_branch.eval()

        with torch.inference_mode():
            specialist_outputs = specialist_branch(image_embeds)

        bundle = generate_scene_tokens(
            specialist_outputs=specialist_outputs,
            top_k_objects=5,
        )
        output_blocks.append(build_output_block(sample_index, sample_token, bundle.tokens))

    output_path.write_text("\n\n".join(output_blocks) + "\n", encoding="utf-8")
    print(f"Saved generated tokens to: {output_path}")


if __name__ == "__main__":
    main()
