# coding=utf-8
# Copyright 2024 MapAnythingLlava3D Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
from typing import List, Optional, Union, Dict
import numpy as np
import torch
from pathlib import Path
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, is_valid_image
from transformers.feature_extraction_utils import BatchFeature

# Constants
from LLaVA_3D.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX
)
from .action_tokenizer import SpatialActionTokenizer

logger = logging.getLogger(__name__)

class MapAnythingLlava3DProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    
    # We allow flexible classes, but default to Siglip + Llama/Gemma
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        # Custom config args
        statistics: Optional[dict] = None,
        bin_policy=None,
        intrinsic_config=None,
        action_config=None,
        num_obs_steps=1,
        obs_delta=1,
        action_chunk_size=1,
        min_sigma=0.0,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        
        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(DEFAULT_IMAGE_TOKEN, normalized=False, special=True)
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            self.image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            tokenizer.image_token = DEFAULT_IMAGE_TOKEN
            tokenizer.image_token_id = self.image_token_id
        else:
            self.image_token_id = tokenizer.image_token_id
        self.image_token_index = self.image_token_id
        
        # 2. Derive Image Sequence Length
        # (Needed to expand <image> token into multiple embeddings)
        if hasattr(image_processor, "image_seq_length"):
            self.image_seq_length = image_processor.image_seq_length
        else:
            # Heuristic calculation: (H // patch) * (W // patch)
            # Default SigLIP: 224 / 14 = 16 -> 16*16 = 256
            h = getattr(image_processor, "size", {}).get("height", 224)
            patch = getattr(image_processor, "patch_size", 14)
            self.image_seq_length = (h // patch) ** 2
            setattr(image_processor, "image_seq_length", self.image_seq_length)

        # 3. Action & Intrinsic Config
        # If not passed in init, try to load from tokenizer config or local file
        # (This logic mimics the original code's fallback)
        self.statistics = statistics or {}
        self.bin_policy = bin_policy
        self.intrinsic_config = intrinsic_config or {}
        self.action_config = action_config or {}
        
        self.dataset_intrinsics = {}
        # Pre-process intrinsics based on image size
        if self.intrinsic_config:
            width = getattr(image_processor, "size", {}).get("width", 224)
            height = getattr(image_processor, "size", {}).get("height", 224)
            for k, v in self.intrinsic_config.items():
                K = torch.tensor(v["intrinsic"]).float()
                # Scale intrinsic matrix to match resized image
                K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
                self.dataset_intrinsics[k] = K

        # 4. Action Tokenizer
        if self.action_config:
            self.action_tokenizer = SpatialActionTokenizer(
                tokenizer=tokenizer,
                num_bins=self.action_config.get("num_bins", 100),
                bin_policy=bin_policy,
                use_spherical=self.action_config.get("use_spherical", False),
                min_sigma=min_sigma
            )
        else:
            self.action_tokenizer = None
            
        self.num_obs_steps = num_obs_steps
        self.action_chunk_size = action_chunk_size

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]] = None,
        images: ImageInput = None,
        unnorm_key: Optional[str] = None,
        suffix_actions: Optional[np.array] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Main entry point for processing.
        1. Process Images -> pixel_values
        2. Expand Text (<image> -> <image> * seq_len)
        3. Tokenize Text -> input_ids
        4. Return combined dict
        """
        
        if text is None and images is None:
            raise ValueError("You must provide either text or images.")
            
        if text is None:
            text = ""

        if isinstance(text, str):
            text = [text]
        
        if images is not None:
            updated = []
            for t in text:
                if isinstance(t, str) and DEFAULT_IMAGE_TOKEN not in t:
                    updated.append(f"{DEFAULT_IMAGE_TOKEN} {t}")
                else:
                    updated.append(t)
            text = updated
        
        # --- 1. Process Images ---
        pixel_values = None
        num_images_per_sample = None
        images_for_processor = images
        if images is not None:
            if isinstance(images, list) and len(images) > 0 and any(isinstance(item, list) for item in images):
                num_images_per_sample = [len(item) if isinstance(item, list) else 1 for item in images]
                images_for_processor = []
                for item in images:
                    if isinstance(item, list):
                        images_for_processor.extend(item)
                    else:
                        images_for_processor.append(item)
            image_outputs = self.image_processor(images_for_processor, return_tensors=return_tensors, **kwargs)
            pixel_values = image_outputs["pixel_values"]
            if num_images_per_sample is not None and isinstance(pixel_values, torch.Tensor):
                b = len(num_images_per_sample)
                if b > 0:
                    if len(set(num_images_per_sample)) == 1:
                        v = num_images_per_sample[0]
                        pixel_values = pixel_values.view(b, v, *pixel_values.shape[1:])
                    else:
                        max_v = max(num_images_per_sample)
                        logger.warning(
                            "Detected varying number of images per sample; padding to max views for batch alignment."
                        )
                        c, h, w = pixel_values.shape[1:]
                        padded = pixel_values.new_zeros((b, max_v, c, h, w))
                        cursor = 0
                        for idx, count in enumerate(num_images_per_sample):
                            if count > 0:
                                padded[idx, :count] = pixel_values[cursor : cursor + count]
                                cursor += count
                        pixel_values = padded
                        num_images_per_sample = [max_v] * b
        
        # --- 2. Expand Text with Image Tokens ---
        # Logic: Replace single <image> token with sequence of <image> tokens
        # to reserve slots for embeddings.
        expanded_text = []
        for idx, t in enumerate(text):
            if DEFAULT_IMAGE_TOKEN in t:
                num_images = 1
                if num_images_per_sample is not None and idx < len(num_images_per_sample):
                    num_images = num_images_per_sample[idx]
                replacement = " ".join([DEFAULT_IMAGE_TOKEN] * (self.image_seq_length * num_images))
                t_expanded = t.replace(DEFAULT_IMAGE_TOKEN, replacement)
                expanded_text.append(t_expanded)
            else:
                expanded_text.append(t)
        
        # --- 3. Tokenize ---
        # Handle suffix actions if provided (for training/conditioning)
        suffix_str = ""
        if suffix_actions is not None and self.action_tokenizer is not None:
            action_tokens = self.action_tokenizer(suffix_actions)
            # Flatten and join
            suffix_str = "".join(action_tokens.flatten().tolist()) # Assuming tokens are strings
            
        # Combine
        final_text = [t + suffix_str for t in expanded_text]
        
        model_inputs = self.tokenizer(
            final_text,
            return_tensors=return_tensors,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", None)
        )
        
        # --- 4. Add Extra Info ---
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            
        # Add intrinsics if available
        if self.intrinsic_config:
            key = unnorm_key if (unnorm_key and unnorm_key in self.dataset_intrinsics) else "default"
            # Fallback to first key if default not found
            if key == "default" and "default" not in self.dataset_intrinsics:
                if len(self.dataset_intrinsics) > 0:
                    key = next(iter(self.dataset_intrinsics))
            
            if key in self.dataset_intrinsics:
                model_inputs["intrinsic"] = self.dataset_intrinsics[key]
                
        # Add constants for model usage
        model_inputs["image_token_index"] = self.image_token_index
        model_inputs["image_token_id"] = self.image_token_id

        return BatchFeature(data=model_inputs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def decode_actions(
        self,
        generation_outputs: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        action_token_num = 3
        predicted_action_token_ids = generation_outputs[0, : action_token_num * self.action_chunk_size].detach().cpu().long().numpy()
        if predicted_action_token_ids.shape[0] < action_token_num * self.action_chunk_size:
            logger.warning("Padding zero action")
            predicted_action_token_ids = np.concatenate(
                [
                    predicted_action_token_ids,
                    np.zeros(action_token_num * self.action_chunk_size - predicted_action_token_ids.shape[0], dtype=np.longlong),
                ]
            )
        predicted_action_token_ids = predicted_action_token_ids.reshape(-1, action_token_num)
        normalized_action_chunks = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids)

        if unnorm_key is None or (unnorm_key not in self.statistics):
            logger.warning(f"unnorm_key {unnorm_key} is not in statistics, fallback to default")
            fallback_key = "default" if ("default" in self.statistics) else next(self.statistics.keys())
            action_norm_stats = self.statistics[fallback_key]["action"]
        else:
            action_norm_stats = self.statistics[unnorm_key]["action"]

        decoded_dim = normalized_action_chunks.shape[1]
        action_dim = len(action_norm_stats["q01"]) if isinstance(action_norm_stats.get("q01"), (list, np.ndarray)) else decoded_dim
        mask_cfg = action_norm_stats.get("mask", np.ones(action_dim))
        mask = np.array(mask_cfg, dtype=bool)
        action_high = np.array(action_norm_stats.get("q99", [1.0] * action_dim))
        action_low = np.array(action_norm_stats.get("q01", [-1.0] * action_dim))
        if action_high.shape[0] != decoded_dim or action_low.shape[0] != decoded_dim or mask.shape[0] != decoded_dim:
            default_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
            default_high = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0])
            default_mask = np.ones_like(default_high, dtype=bool)
            def _fit(arr, default):
                arr = np.array(arr)
                if arr.shape[0] >= decoded_dim:
                    return arr[:decoded_dim]
                else:
                    pad = default[: decoded_dim - arr.shape[0]]
                    return np.concatenate([arr, pad])
            action_low = _fit(action_low, default_low)
            action_high = _fit(action_high, default_high)
            mask = _fit(mask.astype(float), default_mask).astype(bool)

        actions = []
        for normalized_actions in normalized_action_chunks:
            action = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            actions.append(action)
        actions = np.stack(actions)
        return {"actions": actions, "action_ids": predicted_action_token_ids}
