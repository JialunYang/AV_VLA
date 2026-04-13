import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from nuscenes import NuScenes


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENEMMA_ROOT = REPO_ROOT / "OpenEMMA"

for path in (REPO_ROOT, OPENEMMA_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.main_v3 import extract_qwen_image_embeds, load_scene_specialist_branch
from scripts.scene_token_generator import generate_scene_tokens
from training.train_scene_branch import load_frozen_qwen, resolve_qwen_source


DEFAULT_DATAROOT = "datasets/nuscenes"
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_CHECKPOINT = "training/scene_branch/scene_branch.pth"
DEFAULT_OUTPUT = "logs/scene_token_analysis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze generated scene tokens for all nuScenes CAM_FRONT keyframes."
    )
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-model-path", type=str, default=None)
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sample_records(
    nusc: NuScenes,
    dataroot: Path,
) -> List[Tuple[str, Optional[str], str]]:
    records: List[Tuple[str, Optional[str], str]] = []
    for sample in nusc.sample:
        cam_front_token = sample["data"].get("CAM_FRONT")
        if cam_front_token is None:
            continue

        sample_data = nusc.get("sample_data", cam_front_token)
        if not sample_data.get("is_key_frame", False):
            continue

        scene_name: Optional[str] = None
        scene_token = sample.get("scene_token")
        if scene_token:
            scene_name = nusc.get("scene", scene_token)["name"]

        image_path = dataroot / sample_data["filename"]
        records.append((sample["token"], scene_name, str(image_path)))

    return records


def analyze_sample(
    image_path: str,
    processor,
    qwen_model,
    checkpoint_path: Path,
) -> List[str]:
    with torch.inference_mode():
        image_embeds = extract_qwen_image_embeds(image_path, processor, qwen_model)

    specialist_branch = load_scene_specialist_branch(
        input_dim=image_embeds.shape[-1],
        device=image_embeds.device,
        checkpoint_path=checkpoint_path,
    )

    with torch.inference_mode():
        specialist_outputs = specialist_branch(image_embeds)

    bundle = generate_scene_tokens(
        specialist_outputs=specialist_outputs,
        top_k_objects=5,
    )
    return list(bundle.tokens[:7])


def format_debug_block(record: Dict[str, object]) -> str:
    lines = [
        f"sample_token: {record['sample_token']}",
        f"scene_name: {record['scene_name']}",
        f"image_path: {record['image_path']}",
        "tokens:",
    ]
    lines.extend(f"  - {token}" for token in record["tokens"])
    return "\n".join(lines)


def save_results(results: List[Dict[str, object]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


def save_debug_text(results: List[Dict[str, object]], output_file: Path) -> Path:
    debug_path = output_file.with_name(f"{output_file.stem}_debug.txt")
    blocks = [format_debug_block(record) for record in results]
    debug_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return debug_path


def main() -> None:
    args = parse_args()
    dataroot = Path(args.dataroot)
    checkpoint_path = Path(args.checkpoint_path)
    output_file = Path(args.output_file)
    device = resolve_device()

    print(f"Using device: {device}")
    print(f"Loading nuScenes {args.version} from: {dataroot}")
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)

    sample_records = build_sample_records(nusc, dataroot)
    print(f"Found {len(sample_records)} CAM_FRONT keyframe samples.")

    qwen_source = resolve_qwen_source(args.qwen_model_path)
    print(f"Loading Qwen visual model from: {qwen_source}")
    processor, qwen_model = load_frozen_qwen(qwen_source, device)

    results: List[Dict[str, object]] = []
    total_samples = len(sample_records)

    for sample_index, (sample_token, scene_name, image_path) in enumerate(sample_records, start=1):
        tokens = analyze_sample(
            image_path=image_path,
            processor=processor,
            qwen_model=qwen_model,
            checkpoint_path=checkpoint_path,
        )
        results.append(
            {
                "sample_token": sample_token,
                "scene_name": scene_name,
                "image_path": image_path,
                "tokens": tokens,
            }
        )

        if sample_index % 20 == 0 or sample_index == total_samples:
            print(f"Processed {sample_index}/{total_samples} samples")

    save_results(results, output_file)
    debug_path = save_debug_text(results, output_file)

    print(f"Saved JSON results to: {output_file}")
    print(f"Saved debug text to: {debug_path}")


if __name__ == "__main__":
    main()
