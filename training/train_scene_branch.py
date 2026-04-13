import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from nuscenes import NuScenes
from torch.optim import Adam
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on local transformers build
    Qwen2_5_VLForConditionalGeneration = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENEMMA_ROOT = REPO_ROOT / "OpenEMMA"

for path in (REPO_ROOT, OPENEMMA_ROOT, REPO_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.supervision_mapping import (
    CAMERA_CHANNEL,
    NuScenesTableLoader,
    build_proxy_labels_for_sample,
)


DEFAULT_DATAROOT = "datasets/nuscenes"
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "outputs" / "scene_branch"
DEFAULT_QWEN_LOCAL_PATH = REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_EMBEDDINGS_DIR = REPO_ROOT / "training" / "embeddings"

POSITION_TO_ID = {"left": 0, "center": 1, "right": 2}
DEPTH_TO_ID = {"near": 0, "mid": 1, "far": 2}
OCCLUSION_TO_ID = {"visible": 0, "occluded": 1}
LANE_DIRECTION_TO_ID = {"straight": 0, "left": 1, "right": 2}
LANE_PATH_STATE_TO_ID = {"clear": 0, "partially_blocked": 1, "blocked": 2}
RISK_REGION_TO_ID = {"left": 0, "front": 1, "right": 2}
RISK_LEVEL_TO_ID = {"low": 0, "medium": 1, "high": 2}
RISK_SOURCE_TO_ID = {"visible": 0, "hidden": 1}


class SceneSpecialistBranch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.shared_projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.object_class_head = torch.nn.Linear(hidden_dim, num_classes)
        self.object_position_head = torch.nn.Linear(hidden_dim, 3)
        self.object_depth_head = torch.nn.Linear(hidden_dim, 3)
        self.object_occlusion_head = torch.nn.Linear(hidden_dim, 2)
        self.object_importance_head = torch.nn.Linear(hidden_dim, 1)

        self.lane_direction_head = torch.nn.Linear(hidden_dim, 3)
        self.lane_path_state_head = torch.nn.Linear(hidden_dim, 3)

        self.risk_region_head = torch.nn.Linear(hidden_dim, 3)
        self.risk_level_head = torch.nn.Linear(hidden_dim, 3)
        self.risk_source_head = torch.nn.Linear(hidden_dim, 2)

        self.obj_conf_head = torch.nn.Linear(hidden_dim, 1)
        self.lane_conf_head = torch.nn.Linear(hidden_dim, 1)
        self.risk_conf_head = torch.nn.Linear(hidden_dim, 1)

    def _pool_scene_features(self, shared_features):
        return shared_features.mean(dim=0, keepdim=True)

    def forward(self, image_embeds):
        if image_embeds.ndim != 2:
            raise ValueError(
                f"SceneSpecialistBranch expects image_embeds with shape (N, D), got {tuple(image_embeds.shape)}"
            )

        shared_features = self.shared_projection(image_embeds)
        scene_features = self._pool_scene_features(shared_features)

        outputs = {
            "object": {
                "class_logits": self.object_class_head(shared_features),
                "position_logits": self.object_position_head(shared_features),
                "depth_logits": self.object_depth_head(shared_features),
                "occlusion_logits": self.object_occlusion_head(shared_features),
                "importance": self.object_importance_head(shared_features),
            },
            "lane": {
                "direction_logits": self.lane_direction_head(scene_features),
                "path_state_logits": self.lane_path_state_head(scene_features),
            },
            "risk": {
                "region_logits": self.risk_region_head(scene_features),
                "level_logits": self.risk_level_head(scene_features),
                "source_logits": self.risk_source_head(scene_features),
            },
            "confidence": {
                "object_token_confidence": self.obj_conf_head(shared_features),
                "lane_token_confidence": self.lane_conf_head(scene_features),
                "risk_token_confidence": self.risk_conf_head(scene_features),
            },
        }
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Scene Specialist Branch with frozen Qwen visual features."
    )
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION)
    parser.add_argument("--qwen-model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embeddings-dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=23)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_qwen_source(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path

    if DEFAULT_QWEN_LOCAL_PATH.exists():
        return str(DEFAULT_QWEN_LOCAL_PATH)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / "models--Qwen--Qwen2-VL-7B-Instruct"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = repo_dir / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return str(snapshot_path)

    return "Qwen/Qwen2-VL-7B-Instruct"


def load_frozen_qwen(model_path: str, device: torch.device):
    device = torch.device(device)
    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": None,
    }

    if "Qwen2.5" in model_path or "Qwen2_5" in model_path:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "This transformers build does not expose Qwen2_5_VLForConditionalGeneration."
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


class NuScenesSceneBranchDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        version: str,
        camera_channel: str = CAMERA_CHANNEL,
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataroot = dataroot
        self.version = version
        self.camera_channel = camera_channel
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.proxy_loader = NuScenesTableLoader(dataroot=dataroot, version=version)
        self.sample_tokens = self._collect_sample_tokens(max_samples=max_samples)

    def _collect_sample_tokens(self, max_samples: Optional[int]) -> List[str]:
        tokens: List[str] = []
        for sample in self.nusc.sample:
            sample_data = self.nusc.get("sample_data", sample["data"][self.camera_channel])
            if not sample_data.get("is_key_frame", False):
                continue
            tokens.append(sample["token"])
            if max_samples is not None and len(tokens) >= max_samples:
                break
        return tokens

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_token = self.sample_tokens[index]
        sample = self.nusc.get("sample", sample_token)
        sample_data = self.nusc.get("sample_data", sample["data"][self.camera_channel])
        image_path = Path(self.nusc.dataroot) / sample_data["filename"]
        proxy_labels = build_proxy_labels_for_sample(self.proxy_loader, sample_token)
        return {
            "sample_token": sample_token,
            "image_path": str(image_path),
            "proxy_labels": proxy_labels,
        }


class CategoryMapper:
    def __init__(self, max_classes: int) -> None:
        self.max_classes = max_classes
        self.mapping: Dict[str, int] = {}

    def encode(self, category_name: str) -> int:
        if category_name in self.mapping:
            return self.mapping[category_name]

        if len(self.mapping) < self.max_classes:
            class_id = len(self.mapping)
            self.mapping[category_name] = class_id
            return class_id

        fallback_id = self.max_classes - 1
        self.mapping[category_name] = fallback_id
        return fallback_id


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_targets(
    proxy_labels: Dict[str, Any],
    category_mapper: CategoryMapper,
    num_object_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    object_labels = proxy_labels["object_proxy_labels"][:num_object_tokens]
    lane_proxy = proxy_labels["lane_proxy_token"]
    risk_proxy = proxy_labels["risk_proxy_token"]

    object_targets: List[Dict[str, torch.Tensor]] = []
    for obj in object_labels:
        importance = clamp01(obj.get("importance_proxy_score", 0.0))
        confidence = clamp01(0.4 + 0.6 * importance)
        object_targets.append(
            {
                "class": torch.tensor(
                    category_mapper.encode(obj["category_name"]),
                    dtype=torch.long,
                    device=device,
                ),
                "position": torch.tensor(
                    POSITION_TO_ID[obj["horizontal_position"]],
                    dtype=torch.long,
                    device=device,
                ),
                "depth": torch.tensor(
                    DEPTH_TO_ID[obj["depth"]],
                    dtype=torch.long,
                    device=device,
                ),
                "occlusion": torch.tensor(
                    OCCLUSION_TO_ID[obj["occlusion"]],
                    dtype=torch.long,
                    device=device,
                ),
                "importance": torch.tensor([importance], dtype=torch.float32, device=device),
                "confidence": torch.tensor([confidence], dtype=torch.float32, device=device),
            }
        )

    lane_confidence = 0.9 if lane_proxy["path_state"] == "clear" else 0.7
    risk_confidence = clamp01(0.5 + 0.5 * float(risk_proxy.get("score", 0.1)))

    return {
        "objects": object_targets,
        "lane": {
            "direction": torch.tensor(
                LANE_DIRECTION_TO_ID.get(lane_proxy["direction"], 0),
                dtype=torch.long,
                device=device,
            ),
            "path_state": torch.tensor(
                LANE_PATH_STATE_TO_ID.get(lane_proxy["path_state"], 0),
                dtype=torch.long,
                device=device,
            ),
            "confidence": torch.tensor([lane_confidence], dtype=torch.float32, device=device),
        },
        "risk": {
            "region": torch.tensor(
                RISK_REGION_TO_ID.get(risk_proxy["region"], 1),
                dtype=torch.long,
                device=device,
            ),
            "level": torch.tensor(
                RISK_LEVEL_TO_ID.get(risk_proxy["risk_level"], 0),
                dtype=torch.long,
                device=device,
            ),
            "source": torch.tensor(
                RISK_SOURCE_TO_ID.get(risk_proxy["risk_source"], 0),
                dtype=torch.long,
                device=device,
            ),
            "confidence": torch.tensor([risk_confidence], dtype=torch.float32, device=device),
        },
    }


def object_loss(outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    device = outputs["object"]["class_logits"].device
    object_targets = targets["objects"]
    if not object_targets:
        return torch.tensor(0.0, device=device)

    num_prediction_tokens = outputs["object"]["class_logits"].shape[0]
    k = min(5, len(object_targets), num_prediction_tokens)

    class_logits = outputs["object"]["class_logits"][:k]
    position_logits = outputs["object"]["position_logits"][:k]
    depth_logits = outputs["object"]["depth_logits"][:k]
    occlusion_logits = outputs["object"]["occlusion_logits"][:k]
    importance_scores = outputs["object"]["importance"][:k]
    object_confidence_logits = outputs["confidence"]["object_token_confidence"][:k]

    class_targets = torch.stack([target["class"] for target in object_targets[:k]])
    position_targets = torch.stack([target["position"] for target in object_targets[:k]])
    depth_targets = torch.stack([target["depth"] for target in object_targets[:k]])
    occlusion_targets = torch.stack([target["occlusion"] for target in object_targets[:k]])
    importance_targets = torch.stack([target["importance"] for target in object_targets[:k]])
    confidence_targets = torch.stack([target["confidence"] for target in object_targets[:k]])

    class_term = F.cross_entropy(class_logits, class_targets)
    position_term = F.cross_entropy(position_logits, position_targets)
    depth_term = F.cross_entropy(depth_logits, depth_targets)
    occlusion_term = F.cross_entropy(occlusion_logits, occlusion_targets)
    importance_term = F.mse_loss(importance_scores, importance_targets)
    confidence_term = F.binary_cross_entropy_with_logits(
        object_confidence_logits,
        confidence_targets,
    )

    return (
        class_term
        + position_term
        + depth_term
        + occlusion_term
        + importance_term
        + confidence_term
    )


def lane_loss(outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    direction_term = F.cross_entropy(
        outputs["lane"]["direction_logits"],
        targets["lane"]["direction"].unsqueeze(0),
    )
    path_state_term = F.cross_entropy(
        outputs["lane"]["path_state_logits"],
        targets["lane"]["path_state"].unsqueeze(0),
    )
    confidence_term = F.binary_cross_entropy_with_logits(
        outputs["confidence"]["lane_token_confidence"].view(1),
        targets["lane"]["confidence"].view(1),
    )
    return direction_term + path_state_term + confidence_term


def risk_loss(outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    region_term = F.cross_entropy(
        outputs["risk"]["region_logits"],
        targets["risk"]["region"].unsqueeze(0),
    )
    level_term = F.cross_entropy(
        outputs["risk"]["level_logits"],
        targets["risk"]["level"].unsqueeze(0),
    )
    source_term = F.cross_entropy(
        outputs["risk"]["source_logits"],
        targets["risk"]["source"].unsqueeze(0),
    )
    confidence_term = F.binary_cross_entropy_with_logits(
        outputs["confidence"]["risk_token_confidence"].view(1),
        targets["risk"]["confidence"].view(1),
    )
    return region_term + level_term + source_term + confidence_term


def confidence_loss(outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    device = outputs["object"]["class_logits"].device
    confidence_terms = []

    object_targets = targets["objects"]
    if object_targets:
        num_prediction_tokens = outputs["object"]["class_logits"].shape[0]
        k = min(5, len(object_targets), num_prediction_tokens)

        class_predictions = outputs["object"]["class_logits"][:k].argmax(dim=-1)
        position_predictions = outputs["object"]["position_logits"][:k].argmax(dim=-1)
        depth_predictions = outputs["object"]["depth_logits"][:k].argmax(dim=-1)
        occlusion_predictions = outputs["object"]["occlusion_logits"][:k].argmax(dim=-1)
        class_targets = torch.stack([target["class"] for target in object_targets[:k]])
        position_targets = torch.stack([target["position"] for target in object_targets[:k]])
        depth_targets = torch.stack([target["depth"] for target in object_targets[:k]])
        occlusion_targets = torch.stack([target["occlusion"] for target in object_targets[:k]])
        object_correctness = (
            (class_predictions == class_targets)
            & (position_predictions == position_targets)
            & (depth_predictions == depth_targets)
            & (occlusion_predictions == occlusion_targets)
        ).to(torch.float32).unsqueeze(-1)
        object_confidence_logits = outputs["confidence"]["object_token_confidence"][:k]
        confidence_terms.append(
            F.binary_cross_entropy_with_logits(object_confidence_logits, object_correctness)
        )

    lane_direction_correct = (
        outputs["lane"]["direction_logits"].argmax(dim=-1) == targets["lane"]["direction"].view(1)
    ).to(torch.float32)
    lane_path_state_correct = (
        outputs["lane"]["path_state_logits"].argmax(dim=-1) == targets["lane"]["path_state"].view(1)
    ).to(torch.float32)
    lane_correctness = torch.minimum(lane_direction_correct, lane_path_state_correct).view(1)
    confidence_terms.append(
        F.binary_cross_entropy_with_logits(
            outputs["confidence"]["lane_token_confidence"].view(1),
            lane_correctness,
        )
    )

    risk_region_correct = (
        outputs["risk"]["region_logits"].argmax(dim=-1) == targets["risk"]["region"].view(1)
    ).to(torch.float32)
    risk_level_correct = (
        outputs["risk"]["level_logits"].argmax(dim=-1) == targets["risk"]["level"].view(1)
    ).to(torch.float32)
    risk_source_correct = (
        outputs["risk"]["source_logits"].argmax(dim=-1) == targets["risk"]["source"].view(1)
    ).to(torch.float32)
    risk_correctness = torch.minimum(
        risk_region_correct,
        torch.minimum(risk_level_correct, risk_source_correct),
    ).view(1)
    confidence_terms.append(
        F.binary_cross_entropy_with_logits(
            outputs["confidence"]["risk_token_confidence"].view(1),
            risk_correctness,
        )
    )

    if not confidence_terms:
        return torch.tensor(0.0, device=device)
    return torch.stack(confidence_terms).mean()


def compute_loss(outputs: Dict[str, Any], targets: Dict[str, Any]):
    object_term = object_loss(outputs, targets)
    lane_term = lane_loss(outputs, targets)
    risk_term = risk_loss(outputs, targets)
    confidence_term = confidence_loss(outputs, targets)

    total_loss = (
        1.0 * object_term
        + 0.5 * lane_term
        + 0.5 * risk_term
        + 0.5 * confidence_term
    )
    return total_loss, {
        "object_loss": float(object_term.item()),
        "lane_loss": float(lane_term.item()),
        "risk_loss": float(risk_term.item()),
        "confidence_loss": float(confidence_term.item()),
    }


def plot_losses(losses: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Scene Specialist Branch Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    dataset = NuScenesSceneBranchDataset(
        dataroot=args.dataroot,
        version=args.version,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No CAM_FRONT key-frame samples were found in the requested nuScenes split.")

    print(f"Loaded {len(dataset)} samples from nuScenes {args.version}.")

    branch: Optional[SceneSpecialistBranch] = None
    optimizer: Optional[Adam] = None
    category_mapper = CategoryMapper(max_classes=args.num_classes)
    epoch_losses: List[float] = []
    epoch_logs: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if branch is not None:
            branch.train()

        running_loss = 0.0
        running_object_loss = 0.0
        running_lane_loss = 0.0
        running_risk_loss = 0.0
        running_confidence_loss = 0.0
        num_steps = 0

        for sample_index in range(len(dataset)):
            batch = dataset[sample_index]
            sample_token = batch["sample_token"]
            emb_path = embeddings_dir / f"{sample_token}.pt"
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"Missing embedding for {sample_token}. Run precompute_embeddings.py first."
                )
            image_embeds = torch.load(emb_path)
            image_embeds = image_embeds.to(device=device, dtype=torch.float32)

            if branch is None:
                branch = SceneSpecialistBranch(
                    input_dim=image_embeds.shape[-1],
                    hidden_dim=args.hidden_dim,
                    num_classes=args.num_classes,
                ).to(device)
                optimizer = Adam(branch.parameters(), lr=args.lr)

                trainable_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
                print(f"Initialized SceneSpecialistBranch with input_dim={image_embeds.shape[-1]}.")
                print(f"Trainable branch params: {trainable_params}")

            targets = build_targets(
                proxy_labels=batch["proxy_labels"],
                category_mapper=category_mapper,
                num_object_tokens=image_embeds.shape[0],
                device=device,
            )

            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            outputs = branch(image_embeds)
            loss, loss_breakdown = compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_object_loss += loss_breakdown["object_loss"]
            running_lane_loss += loss_breakdown["lane_loss"]
            running_risk_loss += loss_breakdown["risk_loss"]
            running_confidence_loss += loss_breakdown["confidence_loss"]
            num_steps += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if sample_index % 20 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"sample {sample_index + 1}/{len(dataset)} | "
                    f"token={sample_token} | "
                    f"loss={loss.item():.4f} | "
                    f"object={loss_breakdown['object_loss']:.4f} | "
                    f"lane={loss_breakdown['lane_loss']:.4f} | "
                    f"risk={loss_breakdown['risk_loss']:.4f} | "
                    f"confidence={loss_breakdown['confidence_loss']:.4f}"
                )

        epoch_loss = running_loss / max(num_steps, 1)
        epoch_losses.append(epoch_loss)
        epoch_log = {
            "epoch": epoch,
            "total_loss": epoch_loss,
            "object_loss": running_object_loss / max(num_steps, 1),
            "lane_loss": running_lane_loss / max(num_steps, 1),
            "risk_loss": running_risk_loss / max(num_steps, 1),
            "confidence_loss": running_confidence_loss / max(num_steps, 1),
        }
        epoch_logs.append(epoch_log)
        print(
            f"Epoch {epoch}/{args.epochs} average loss: {epoch_log['total_loss']:.4f} | "
            f"object={epoch_log['object_loss']:.4f} | "
            f"lane={epoch_log['lane_loss']:.4f} | "
            f"risk={epoch_log['risk_loss']:.4f} | "
            f"confidence={epoch_log['confidence_loss']:.4f}"
        )

    if branch is None:
        raise RuntimeError("Training did not initialize the SceneSpecialistBranch.")

    checkpoint_path = output_dir / "scene_branch.pth"
    state_dict = branch.state_dict()
    conf_keys = [key for key in state_dict.keys() if "conf" in key]
    print(conf_keys)
    unexpected_conf_keys = [
        key for key in conf_keys
        if not key.startswith(("obj_conf_head.", "lane_conf_head.", "risk_conf_head."))
    ]
    if unexpected_conf_keys:
        raise RuntimeError(f"Refusing to save checkpoint with unexpected confidence keys: {unexpected_conf_keys}")

    torch.save(
        {
            "model_state_dict": state_dict,
            "input_dim": branch.input_dim,
            "hidden_dim": branch.hidden_dim,
            "num_classes": branch.num_classes,
            "category_mapping": category_mapper.mapping,
            "epoch_losses": epoch_losses,
            "epoch_logs": epoch_logs,
            "version": args.version,
            "dataroot": args.dataroot,
        },
        checkpoint_path,
    )

    loss_curve_path = output_dir / "loss_curve.png"
    plot_losses(epoch_losses, loss_curve_path)

    log_path = output_dir / "training_log.json"
    log_path.write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "num_samples": len(dataset),
                "epoch_losses": epoch_losses,
                "epoch_logs": epoch_logs,
                "checkpoint": str(checkpoint_path),
                "loss_curve": str(loss_curve_path),
                "embeddings_dir": str(embeddings_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved training log to: {log_path}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
