from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


POSITION_LABELS = ("left", "center", "right")
DEPTH_LABELS = ("near", "mid", "far")
OCCLUSION_LABELS = ("visible", "occluded")
LANE_DIRECTION_LABELS = ("straight", "left", "right")
LANE_PATH_STATE_LABELS = ("clear", "partially_blocked", "blocked")
RISK_REGION_LABELS = ("left", "front", "right")
RISK_LEVEL_LABELS = ("low", "medium", "high")
RISK_SOURCE_LABELS = ("visible", "hidden")


@dataclass(frozen=True)
class SceneTokenBundle:
    structured: Dict[str, object]
    tokens: List[str]


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _to_array(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach()
    return np.asarray(value)


def _sanitize_finite(value: Any, default: float = 0.0) -> Any:
    tensor = _to_array(value)
    if _is_torch_tensor(tensor):
        return torch.nan_to_num(tensor, nan=default, posinf=default, neginf=default)
    return np.nan_to_num(tensor, nan=default, posinf=default, neginf=default)


def _squeeze_last_dim(value: Any) -> Any:
    tensor = _sanitize_finite(value)
    if tensor.ndim > 0 and tensor.shape[-1] == 1:
        if _is_torch_tensor(tensor):
            return tensor.squeeze(-1)
        return np.squeeze(tensor, axis=-1)
    return tensor


def _sigmoid(value: Any) -> Any:
    tensor = _sanitize_finite(value)
    if _is_torch_tensor(tensor):
        return torch.sigmoid(tensor)
    return 1.0 / (1.0 + np.exp(-tensor))


def _scalar_sigmoid(value: Any) -> float:
    tensor = _sanitize_finite(value).reshape(-1)
    return float(_sanitize_finite(_sigmoid(tensor))[0])


def _argmax_index(logits: Any) -> int:
    tensor = _sanitize_finite(logits)
    if _is_torch_tensor(tensor):
        return int(torch.argmax(tensor, dim=-1).item())
    return int(np.argmax(tensor, axis=-1).item())


def _argmax_label(logits: Any, labels: Sequence[str]) -> Tuple[int, str]:
    index = _argmax_index(logits)
    return index, labels[index]


def _format_score(value: float) -> str:
    value = float(_sanitize_finite(np.asarray(value)).item())
    return f"{value:.4f}"


def build_object_candidates(
    specialist_outputs: Dict[str, Dict[str, Any]],
    object_class_labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    object_outputs = specialist_outputs["object"]
    confidence_outputs = specialist_outputs["confidence"]

    class_logits = _sanitize_finite(object_outputs["class_logits"])
    position_logits = _sanitize_finite(object_outputs["position_logits"])
    depth_logits = _sanitize_finite(object_outputs["depth_logits"])
    occlusion_logits = _sanitize_finite(object_outputs["occlusion_logits"])
    importance_logits = _squeeze_last_dim(object_outputs["importance"])
    confidence_logits = _squeeze_last_dim(confidence_outputs["object_token_confidence"])

    num_candidates = class_logits.shape[0]
    num_classes = class_logits.shape[-1]
    class_labels = list(object_class_labels or [f"class_{idx}" for idx in range(num_classes)])

    if len(class_labels) < num_classes:
        raise ValueError(
            f"Need at least {num_classes} object class labels, got {len(class_labels)}."
        )

    candidates: List[Dict[str, object]] = []
    for token_index in range(num_candidates):
        class_index = _argmax_index(class_logits[token_index])
        position_index = _argmax_index(position_logits[token_index])
        depth_index = _argmax_index(depth_logits[token_index])
        occlusion_index = _argmax_index(occlusion_logits[token_index])

        importance_score = float(_sanitize_finite(_sigmoid(importance_logits[token_index])).item())
        confidence_score = float(_sanitize_finite(_sigmoid(confidence_logits[token_index])).item())

        candidates.append(
            {
                "token_index": token_index,
                "class_index": class_index,
                "class": class_labels[class_index],
                "pos_index": position_index,
                "pos": POSITION_LABELS[position_index],
                "depth_index": depth_index,
                "depth": DEPTH_LABELS[depth_index],
                "occ_index": occlusion_index,
                "occ": OCCLUSION_LABELS[occlusion_index],
                "importance": importance_score,
                "conf": confidence_score,
            }
        )

    return candidates


def rank_object_candidates(
    candidates: Iterable[Dict[str, object]],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["importance"]),
            float(candidate["conf"]),
            -int(candidate["token_index"]),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _format_object_token(candidate: Dict[str, object]) -> str:
    return (
        "[OBJ] "
        f"class={candidate['class']} "
        f"pos={candidate['pos']} "
        f"depth={candidate['depth']} "
        f"occ={candidate['occ']} "
        f"importance={_format_score(float(candidate['importance']))} "
        f"conf={_format_score(float(candidate['conf']))}"
    )


def _build_lane_token(specialist_outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, object]:
    lane_outputs = specialist_outputs["lane"]
    confidence_outputs = specialist_outputs["confidence"]

    direction_index, direction = _argmax_label(
        lane_outputs["direction_logits"][0], LANE_DIRECTION_LABELS
    )
    path_state_index, path_state = _argmax_label(
        lane_outputs["path_state_logits"][0], LANE_PATH_STATE_LABELS
    )
    confidence = _scalar_sigmoid(confidence_outputs["lane_token_confidence"][0])

    return {
        "direction_index": direction_index,
        "direction": direction,
        "path_state_index": path_state_index,
        "path_state": path_state,
        "conf": confidence,
    }


def _format_lane_token(lane_token: Dict[str, object]) -> str:
    return (
        "[LANE] "
        f"direction={lane_token['direction']} "
        f"path_state={lane_token['path_state']} "
        f"conf={_format_score(float(lane_token['conf']))}"
    )


def _build_risk_token(specialist_outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, object]:
    risk_outputs = specialist_outputs["risk"]
    confidence_outputs = specialist_outputs["confidence"]

    region_index, region = _argmax_label(risk_outputs["region_logits"][0], RISK_REGION_LABELS)
    level_index, level = _argmax_label(risk_outputs["level_logits"][0], RISK_LEVEL_LABELS)
    source_index, source = _argmax_label(risk_outputs["source_logits"][0], RISK_SOURCE_LABELS)
    confidence = _scalar_sigmoid(confidence_outputs["risk_token_confidence"][0])

    return {
        "region_index": region_index,
        "region": region,
        "level_index": level_index,
        "level": level,
        "source_index": source_index,
        "source": source,
        "conf": confidence,
    }


def _format_risk_token(risk_token: Dict[str, object]) -> str:
    return (
        "[RISK] "
        f"region={risk_token['region']} "
        f"level={risk_token['level']} "
        f"source={risk_token['source']} "
        f"conf={_format_score(float(risk_token['conf']))}"
    )


def generate_scene_tokens(
    specialist_outputs: Dict[str, Dict[str, Any]],
    object_class_labels: Optional[Sequence[str]] = None,
    top_k_objects: int = 5,
) -> SceneTokenBundle:
    candidates = build_object_candidates(
        specialist_outputs=specialist_outputs,
        object_class_labels=object_class_labels,
    )
    top_objects = rank_object_candidates(candidates, top_k=top_k_objects)
    lane_token = _build_lane_token(specialist_outputs)
    risk_token = _build_risk_token(specialist_outputs)

    tokens = [_format_object_token(candidate) for candidate in top_objects]
    tokens.append(_format_lane_token(lane_token))
    tokens.append(_format_risk_token(risk_token))

    structured = {
        "object_candidates": candidates,
        "top_objects": top_objects,
        "lane": lane_token,
        "risk": risk_token,
    }
    return SceneTokenBundle(structured=structured, tokens=tokens)


def save_tokens(tokens: Sequence[str], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return output_path


def _build_dummy_outputs(
    num_visual_tokens: int = 12,
    num_classes: int = 6,
    seed: int = 7,
) -> Dict[str, Dict[str, Any]]:
    if torch is not None:
        generator = torch.Generator().manual_seed(seed)
        randn = lambda *shape: torch.randn(*shape, generator=generator)
    else:
        generator = np.random.default_rng(seed)
        randn = lambda *shape: generator.standard_normal(size=shape)

    return {
        "object": {
            "class_logits": randn(num_visual_tokens, num_classes),
            "position_logits": randn(num_visual_tokens, 3),
            "depth_logits": randn(num_visual_tokens, 3),
            "occlusion_logits": randn(num_visual_tokens, 2),
            "importance": randn(num_visual_tokens, 1),
        },
        "lane": {
            "direction_logits": randn(1, 3),
            "path_state_logits": randn(1, 3),
        },
        "risk": {
            "region_logits": randn(1, 3),
            "level_logits": randn(1, 3),
            "source_logits": randn(1, 2),
        },
        "confidence": {
            "object_token_confidence": randn(num_visual_tokens, 1),
            "lane_token_confidence": randn(1, 1),
            "risk_token_confidence": randn(1, 1),
        },
    }


def demo() -> SceneTokenBundle:
    dummy_outputs = _build_dummy_outputs()
    object_labels = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
    bundle = generate_scene_tokens(dummy_outputs, object_class_labels=object_labels, top_k_objects=5)

    output_path = Path(__file__).resolve().parent / "example_scene_tokens.txt"
    save_tokens(bundle.tokens, output_path)

    print("Structured output:")
    print(bundle.structured)
    print("\nGenerated tokens:")
    for token in bundle.tokens:
        print(token)
    print(f"\nSaved example tokens to: {output_path}")

    return bundle


if __name__ == "__main__":
    demo()
