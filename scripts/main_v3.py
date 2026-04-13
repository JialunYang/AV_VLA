import base64
import os.path
import re
import argparse
import sys
from datetime import datetime
from math import atan2
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from openai import OpenAI
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import json

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENEMMA_ROOT = REPO_ROOT / "OpenEMMA"

for path in (REPO_ROOT, OPENEMMA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from transformers import MllamaForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
from scene_token_generator import generate_scene_tokens

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on local transformers version
    Qwen2_5_VLForConditionalGeneration = None

client = None

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN
DEFAULT_SCENE_TOKEN_FILE = Path(__file__).resolve().parent.parent / "tokens_20_samples.txt"
DEFAULT_SCENE_BRANCH_CHECKPOINT = REPO_ROOT / "training" / "scene_branch" / "scene_branch.pth"
_SCENE_TOKEN_CACHE = None
_SCENE_BRANCH_CACHE = None
DEFAULT_DEBUG_DIR = REPO_ROOT / "qwen_results" / "scene_token_debug"


def load_scene_token_map(token_file=DEFAULT_SCENE_TOKEN_FILE):
    global _SCENE_TOKEN_CACHE

    token_path = Path(token_file)
    if _SCENE_TOKEN_CACHE is not None and _SCENE_TOKEN_CACHE["path"] == token_path:
        return _SCENE_TOKEN_CACHE["tokens"]

    if not token_path.exists():
        _SCENE_TOKEN_CACHE = {"path": token_path, "tokens": {}}
        return _SCENE_TOKEN_CACHE["tokens"]

    token_map = {}
    current_sample_token = None
    current_lines = []

    for raw_line in token_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Sample ") and ":" in line:
            if current_sample_token is not None and current_lines:
                token_map[current_sample_token] = current_lines

            current_sample_token = line.split(":", 1)[1].strip()
            current_lines = []
            continue

        if current_sample_token is not None:
            current_lines.append(line)

    if current_sample_token is not None and current_lines:
        token_map[current_sample_token] = current_lines

    _SCENE_TOKEN_CACHE = {"path": token_path, "tokens": token_map}
    return token_map


def build_scene_summary_prefix(sample_token, token_file=DEFAULT_SCENE_TOKEN_FILE):
    if not sample_token:
        return None

    scene_token_lines = load_scene_token_map(token_file=token_file).get(sample_token)
    if not scene_token_lines:
        return None

    expected_prefixes = ["[OBJ]"] * 5 + ["[LANE]", "[RISK]"]
    if len(scene_token_lines) < len(expected_prefixes):
        return None

    formatted_lines = scene_token_lines[: len(expected_prefixes)]
    if any(
        not line.startswith(expected_prefix)
        for line, expected_prefix in zip(formatted_lines, expected_prefixes)
    ):
        return None

    return "Scene summary:\n" + "\n".join(formatted_lines) + "\n\n"


def normalize_visual_tokens(image_embeds):
    if image_embeds is None:
        raise ValueError("Expected non-empty image embeddings for SceneSpecialistBranch.")

    if image_embeds.ndim == 5 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds.squeeze(0)

    if image_embeds.ndim == 4 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds.squeeze(0)

    if image_embeds.ndim > 2:
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

    if image_embeds.ndim != 2:
        raise ValueError(
            f"Expected normalized image_embeds with shape (N, D), got {tuple(image_embeds.shape)}"
        )

    return image_embeds


def extract_qwen_image_embeds(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract scene-specialist visual tokens."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    visual_model = getattr(model, "visual", None)
    if visual_model is None and hasattr(model, "model"):
        visual_model = getattr(model.model, "visual", None)

    if visual_model is None:
        raise AttributeError("Qwen model does not expose a visual encoder for image embeddings.")

    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            image_embeds = model.get_image_features(
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        else:
            try:
                image_embeds = visual_model(
                    inputs.get("pixel_values"),
                    grid_thw=inputs.get("image_grid_thw"),
                )
            except TypeError:
                image_embeds = visual_model(
                    inputs.get("pixel_values"),
                    inputs.get("image_grid_thw"),
                )

    if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]

    return normalize_visual_tokens(image_embeds.float())


def load_scene_specialist_branch(input_dim, device, checkpoint_path=DEFAULT_SCENE_BRANCH_CHECKPOINT):
    global _SCENE_BRANCH_CACHE

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SceneSpecialistBranch checkpoint not found at {checkpoint_path}"
        )

    cache_key = (str(checkpoint_path.resolve()), str(device), int(input_dim))
    if _SCENE_BRANCH_CACHE is not None and _SCENE_BRANCH_CACHE["key"] == cache_key:
        return _SCENE_BRANCH_CACHE["model"]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    specialist_branch = SceneSpecialistBranch(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=18,
    ).to(device)
    specialist_branch.load_state_dict(checkpoint["model_state_dict"])
    specialist_branch.eval()
    print("Loaded SceneSpecialistBranch checkpoint successfully")

    _SCENE_BRANCH_CACHE = {
        "key": cache_key,
        "model": specialist_branch,
    }
    return specialist_branch


def build_runtime_scene_summary(
    sample_token,
    image_path,
    processor,
    model,
    scene_id=None,
    debug_dir=None,
    checkpoint_path=DEFAULT_SCENE_BRANCH_CHECKPOINT,
):
    image_embeds = extract_qwen_image_embeds(image_path, processor, model)
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
    if len(bundle.tokens) < 7:
        raise ValueError(f"Expected 7 scene tokens, got {len(bundle.tokens)}")

    prefix = "Scene summary:\n" + "\n".join(bundle.tokens[:7]) + "\n\n"
    debug_payload = {
        "sample_token": sample_token,
        "image_embeds_shape": list(image_embeds.shape),
        "scene_tokens": bundle.tokens[:7],
        "prompt_prefix": prefix,
    }

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        file_id = scene_id if scene_id is not None else sample_token
        debug_path = debug_dir / f"{file_id}_scene_summary.json"
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        print(f"Saved scene-token debug to {debug_path}")

    print(f"SceneSpecialistBranch ran for sample {sample_token} with image_embeds shape {tuple(image_embeds.shape)}")
    return prefix, debug_payload


def save_prompt_and_output_debug(sample_token, prompt, result, debug_dir, scene_id=None):
    if debug_dir is None:
        return

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    file_id = scene_id if scene_id is not None else sample_token
    debug_path = debug_dir / f"{file_id}_prompt_debug.txt"
    prompt_snippet = prompt[:2000]
    result_snippet = (result or "")[:1000]
    debug_text = (
        f"Sample token: {sample_token}\n\n"
        f"Final prompt snippet:\n{prompt_snippet}\n\n"
        f"Generated output snippet:\n{result_snippet}\n"
    )
    debug_path.write_text(debug_text, encoding="utf-8")
    print(f"Saved prompt/output debug to {debug_path}")


def resolve_cached_hf_snapshot(model_name):
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = repo_dir / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return snapshot_path
    return None


class SceneSpecialistBranch(nn.Module):
    """Lightweight scene-specialist branch over visual token features.

    Input:
        image_embeds: Tensor of shape (N, D)
            N = number of visual tokens
            D = visual feature dimension

    Output:
        Dictionary of logits/scores for object, lane, risk, and confidence heads.
        Object predictions are token-level, while lane and risk predictions are
        scene-level and therefore use a pooled scene feature.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Shared projection used by all specialist heads.
        self.shared_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Object head.
        self.object_class_head = nn.Linear(hidden_dim, num_classes)
        self.object_position_head = nn.Linear(hidden_dim, 3)
        self.object_depth_head = nn.Linear(hidden_dim, 3)
        self.object_occlusion_head = nn.Linear(hidden_dim, 2)
        self.object_importance_head = nn.Linear(hidden_dim, 1)

        # Lane head.
        self.lane_direction_head = nn.Linear(hidden_dim, 3)
        self.lane_path_state_head = nn.Linear(hidden_dim, 3)

        # Risk head.
        self.risk_region_head = nn.Linear(hidden_dim, 3)
        self.risk_level_head = nn.Linear(hidden_dim, 3)
        self.risk_source_head = nn.Linear(hidden_dim, 2)

        # Confidence heads.
        self.obj_conf_head = nn.Linear(hidden_dim, 1)
        self.lane_conf_head = nn.Linear(hidden_dim, 1)
        self.risk_conf_head = nn.Linear(hidden_dim, 1)

    def _pool_scene_features(self, shared_features):
        """Pool token features into one scene feature for scene-level heads."""
        return shared_features.mean(dim=0, keepdim=True)

    def forward(self, image_embeds):
        """Run the specialist heads on visual token features.

        Args:
            image_embeds: Tensor with shape (N, D).

        Returns:
            dict containing:
                object:
                    class_logits: (N, num_classes)
                    position_logits: (N, 3) for left / center / right
                    depth_logits: (N, 3) for near / mid / far
                    occlusion_logits: (N, 2) for visible / occluded
                    importance: (N, 1) scalar importance score
                lane:
                    direction_logits: (1, 3) for straight / left / right
                    path_state_logits: (1, 3) for clear / partially_blocked / blocked
                risk:
                    region_logits: (1, 3) for left / front / right
                    level_logits: (1, 3) for low / medium / high
                    source_logits: (1, 2) for visible / hidden
                confidence:
                    object_token_confidence: (N, 1) scalar confidence per object candidate
                    lane_token_confidence: (1, 1) scalar confidence for the lane token
                    risk_token_confidence: (1, 1) scalar confidence for the risk token
        """
        if image_embeds.ndim != 2:
            raise ValueError(
                f"SceneSpecialistBranch expects image_embeds with shape (N, D), got {tuple(image_embeds.shape)}"
            )

        shared_features = self.shared_projection(image_embeds)
        scene_features = self._pool_scene_features(shared_features)

        outputs = {
            "object": {
                # Object category logits for each visual token.
                "class_logits": self.object_class_head(shared_features),
                # Relative horizontal position logits: left, center, right.
                "position_logits": self.object_position_head(shared_features),
                # Relative depth logits: near, mid, far.
                "depth_logits": self.object_depth_head(shared_features),
                # Occlusion logits: visible or occluded.
                "occlusion_logits": self.object_occlusion_head(shared_features),
                # Scalar importance score used for ranking token candidates.
                "importance": self.object_importance_head(shared_features),
            },
            "lane": {
                # Lane direction logits from one pooled scene representation.
                "direction_logits": self.lane_direction_head(scene_features),
                # Path-state logits from one pooled scene representation.
                "path_state_logits": self.lane_path_state_head(scene_features),
            },
            "risk": {
                # Risk region logits from one pooled scene representation.
                "region_logits": self.risk_region_head(scene_features),
                # Risk level logits from one pooled scene representation.
                "level_logits": self.risk_level_head(scene_features),
                # Risk source logits from one pooled scene representation.
                "source_logits": self.risk_source_head(scene_features),
            },
            "confidence": {
                # Scalar confidence for each object token candidate.
                "object_token_confidence": self.obj_conf_head(shared_features),
                # Scalar confidence for the pooled lane token candidate.
                "lane_token_confidence": self.lane_conf_head(scene_features),
                # Scalar confidence for the pooled risk token candidate.
                "risk_token_confidence": self.risk_conf_head(scene_features),
            },
        }

        return outputs

def getMessage(prompt, image=None, args=None):
    if "llama" in args.model_path or "Llama" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]   
    return message


def get_openai_client():
    global client
    if client is None:
        client = OpenAI()
    return client


def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None):
        if "llama" in args.model_path or "Llama" in args.model_path:
            image = Image.open(images).convert('RGB')
            message = getMessage(text, args=args)
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(**inputs, max_new_tokens=2048)

            output_text = processor.decode(output[0])

            if "llama" in args.model_path or "Llama" in args.model_path:
                output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
            return output_text
        
        elif "qwen" in args.model_path or "Qwen" in args.model_path:
            message = getMessage(text, image=images, args=args)
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

        elif "llava" in args.model_path:
            conv_mode = "mistral_instruct"
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in text:
                if model.config.mm_use_im_start_end:
                    text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
                else:
                    text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
            else:
                if model.config.mm_use_im_start_end:
                    text = image_token_se + "\n" + text
                else:
                    text = DEFAULT_IMAGE_TOKEN + "\n" + text

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image = Image.open(images).convert('RGB')

            image_tensor = process_images([image], processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=2048,
                    use_cache=True,
                    pad_token_id = tokenizer.eos_token_id,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs
                    
        elif "gpt" in args.model_path:
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        *map(lambda x: {"image": x, "resize": 768}, images),
                        text,
                    ],
                },
            ]
            if sys_message is not None:
                sys_message_dict = {
                    "role": "system",
                    "content": sys_message
                }
                PROMPT_MESSAGES.append(sys_message_dict)
            params = {
                "model": "gpt-4o-2024-11-20",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 400,
            }

            result = get_openai_client().chat.completions.create(**params)

            return result.choices[0].message.content

def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    if "llava" in args.model_path:
        prompt = f"""You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):

    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result

def DescribeOrUpdateIntent(obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):

    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
        
    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result


def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, sample_token=None, scene_id=None, processor=None, model=None, tokenizer=None, args=None, debug_dir=None):
    # assert len(obs_images) == len(obs_waypoints)

    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
        print(f'Scene Description: {scene_description}')
        print(f'Object Description: {object_description}')
        print(f'Intent Description: {intent_description}')

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

    if args.method == "openemma":
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
        scene_summary_prefix = None
        if (
            sample_token is not None
            and isinstance(obs_images, str)
            and ("qwen" in args.model_path or "Qwen" in args.model_path)
            and processor is not None
            and model is not None
        ):
            try:
                scene_summary_prefix, _ = build_runtime_scene_summary(
                    sample_token=sample_token,
                    image_path=obs_images,
                    processor=processor,
                    model=model,
                    scene_id=scene_id,
                    debug_dir=debug_dir,
                    checkpoint_path=args.scene_branch_checkpoint,
                )
            except Exception as exc:
                print(f"SceneSpecialistBranch runtime generation failed for {sample_token}: {exc}")

        if scene_summary_prefix is None:
            scene_summary_prefix = build_scene_summary_prefix(sample_token)
            if scene_summary_prefix is not None:
                print(f"Loaded Scene summary from token file for sample {sample_token}")

        if scene_summary_prefix is not None:
            prompt = scene_summary_prefix + prompt
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer, args=args)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    save_prompt_and_output_debug(sample_token, prompt, result, debug_dir, scene_id=scene_id)
    return result, scene_description, object_description, intent_description

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/nuscenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--method", type=str, default='openemma')
    parser.add_argument(
        "--scene-branch-checkpoint",
        type=str,
        default=str(DEFAULT_SCENE_BRANCH_CHECKPOINT),
    )
    args = parser.parse_args()

    print(f"{args.model_path}")

    model = None
    processor = None
    tokenizer = None
    qwen25_loaded = False
    try:
        # 优先本地加载Qwen2.5-VL-3B-Instruct，并优选flash attention
        if "qwen" in args.model_path or "Qwen" in args.model_path:
            local_qwen25_candidates = [
                REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
                Path(args.model_path),
            ]
            local_qwen25_path = next(
                (candidate for candidate in local_qwen25_candidates if candidate.exists()),
                None,
            )
            try:
                if Qwen2_5_VLForConditionalGeneration is None:
                    raise ImportError("Installed transformers build does not expose Qwen2_5_VLForConditionalGeneration")
                if local_qwen25_path is None:
                    raise FileNotFoundError("No local Qwen2.5-VL-3B-Instruct path found under current project layout")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(local_qwen25_path),
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(str(local_qwen25_path))
                tokenizer = None
                qwen25_loaded = True
                print(f"已本地加载 Qwen2.5-VL-3B-Instruct: {local_qwen25_path}")
            except Exception as e:
                print("Qwen2.5-VL-3B-Instruct 加载失败，尝试加载 Qwen2-VL-7B-Instruct。")
                print(e)
                qwen2_vl_source = resolve_cached_hf_snapshot("Qwen/Qwen2-VL-7B-Instruct")
                if qwen2_vl_source is None:
                    qwen2_vl_source = "Qwen/Qwen2-VL-7B-Instruct"
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(qwen2_vl_source),
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(str(qwen2_vl_source))
                tokenizer = None
                qwen25_loaded = False
                print(f"已加载 Qwen2-VL-7B-Instruct: {qwen2_vl_source}")
        else:
            if "llava" == args.model_path:    
                disable_torch_init()
                tokenizer, model, processor, context_len = load_pretrained_model("liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b")
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            elif "llava" in args.model_path:
                disable_torch_init()
                tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, "llava-v1.6-mistral-7b")
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                model = None
                processor = None
                tokenizer=None
    except Exception as e:
        print("模型加载出现异常：", e)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    # Load the dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)

    # Iterate the scenes
    scenes = nusc.scene
    
    print(f"Number of scenes: {len(scenes)}")

    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        description = scene['description']

        if not name in ["scene-0103", "scene-1077"]:
            continue

        # Get all image and pose in this scene
        front_camera_images = []
        sample_tokens = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)
            sample_tokens.append(sample['token'])

            # Get the front camera image of the sample.
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            # nusc.render_sample_data(cam_front_data['token'])


            if "gpt" in args.model_path:
                with open(os.path.join(nusc.dataroot, cam_front_data['filename']), "rb") as image_file:
                    front_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            else:
                front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))

            # Get the ego pose of the sample.
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)

            # Get the camera parameters of the sample.
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            # Advance the pointer.
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ## Compute interpolated trajectory.
        # Get the velocities of the ego vehicle.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # Get the curvature of the ego vehicle.
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],
                                                       atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)

        # Debug
        if args.plot:
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1],
                    color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        # Get the waypoints of the ego vehicle.
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        for i in range(scene_length - TTL_LEN):
            # Get the raw image data.
            # utils.PlotBase64Image(front_camera_images[0])
            obs_images = front_camera_images[i:i+OBS_LEN]
            obs_ego_poses = ego_poses[i:i+OBS_LEN]
            obs_camera_params = camera_params[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]
            current_sample_token = sample_tokens[i + OBS_LEN - 1]
            scene_id = f"{name}_{i}"

            # Get positions of the vehicle.
            obs_start_world = obs_ego_traj_world[0]
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            # obs_images = [curr_image]

            # Allocate the images.
            if "gpt" in args.model_path:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
                img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            else:
                with open(os.path.join(curr_image), "rb") as image_file:
                    img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

            for rho in range(3):
                # Assemble the prompt.
                if not "gpt" in args.model_path:
                    obs_images = curr_image
                (prediction,
                scene_description,
                object_description,
                updated_intent) = GenerateMotion(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                                obs_ego_curvatures, prev_intent, sample_token=current_sample_token, scene_id=scene_id, processor=processor, model=model, tokenizer=tokenizer, args=args, debug_dir=Path(timestamp) / "scene_token_debug")

                # Process the output.
                prev_intent = updated_intent  # Stateful intent
                pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if not coordinates == []:
                    break
            if coordinates == []:
                continue
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # GT
            # OverlayTrajectory(img, fut_ego_traj_world, obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0))

            # Pred
            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,
                                                                   pred_speeds,
                                                                   fut_start_world,
                                                                   atan2(obs_ego_velocities[-1][1],
                                                                         obs_ego_velocities[-1][0]), pred_len)

            # Overlay the trajectory.
            check_flag = OverlayTrajectory(img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args)
            

            # Compute ADE.
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
            
            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len] , axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len] , axis=1))
            ade3s_list.append(ade3s)

            # Write to image.
            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                # Plot the trajectory.
                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                # Save the trajectory
                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                # Save the descriptions
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Scene Description: {scene_description}\n")
                    f.write(f"Object Description: {object_description}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            # break  # Timestep

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade
        }

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

        # break  # Scenes


def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None):
    if ("qwen" in args.model_path or "Qwen" in args.model_path):
        # 判断是否为Qwen2.5-VL-3B-Instruct（新版）
        if hasattr(model, "model_type") and getattr(model, "model_type", "") == "qwen2_5_vl":
            # Qwen2.5-VL-3B-Instruct官方推荐推理方式
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            text_prompt = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        else:
            # 兼容Qwen2-VL-7B-Instruct等老模型
            message = getMessage(text, image=images, args=args)
            text_prompt = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
    # ... 其它模型推理逻辑保持不变 ...
