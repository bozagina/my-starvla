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


def _is_rank0():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


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
        image_token_joiner: str = "auto",
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
        self.image_token_joiner_mode = str(image_token_joiner).strip().lower() if image_token_joiner is not None else "auto"
        self.image_token_joiner = self._resolve_image_token_joiner(tokenizer, self.image_token_id, self.image_token_joiner_mode)
        
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
        self._cot_mask_log_counter = 0
        self._cot_mask_log_first_n = 3

    @staticmethod
    def _extract_ids(tokenizer_output):
        ids = None
        if hasattr(tokenizer_output, "get"):
            ids = tokenizer_output.get("input_ids", None)
        if ids is None and hasattr(tokenizer_output, "input_ids"):
            ids = tokenizer_output.input_ids
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if isinstance(ids, tuple):
            ids = list(ids)
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list):
            return None
        try:
            return [int(x) for x in ids]
        except Exception:
            return None

    @classmethod
    def _resolve_image_token_joiner(cls, tokenizer, image_token_id: int, mode: str) -> str:
        if mode in ("space", "spaced", " "):
            return " "
        if mode in ("empty", "none", "nospace", ""):
            return ""

        token = getattr(tokenizer, "image_token", DEFAULT_IMAGE_TOKEN)
        probe = f"{token}{token}"
        try:
            out = tokenizer(
                probe,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = cls._extract_ids(out)
            if ids is not None and len(ids) == 2 and all(int(x) == int(image_token_id) for x in ids):
                logger.info("MapAnythingLlava3DProcessor: auto-selected empty image token joiner.")
                return ""
        except Exception:
            pass

        logger.info("MapAnythingLlava3DProcessor: auto-selected space image token joiner.")
        return " "

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]] = None,
        images: ImageInput = None,
        unnorm_key: Optional[str] = None,
        instruction_char_spans: Optional[List] = None,
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

        sample_count = len(text)
        span_list = None
        if instruction_char_spans is not None:
            if len(instruction_char_spans) != sample_count:
                logger.warning(
                    "instruction_char_spans length mismatch: spans=%d text=%d, ignore spans.",
                    len(instruction_char_spans),
                    sample_count,
                )
            else:
                span_list = []
                for span in instruction_char_spans:
                    try:
                        s, e = int(span[0]), int(span[1])
                    except Exception:
                        s, e = 0, 0
                    span_list.append([s, e])
        
        if images is not None:
            updated = []
            prefix = f"{DEFAULT_IMAGE_TOKEN} "
            for idx, t in enumerate(text):
                if isinstance(t, str) and DEFAULT_IMAGE_TOKEN not in t:
                    updated.append(prefix + t)
                    if span_list is not None:
                        span_list[idx][0] += len(prefix)
                        span_list[idx][1] += len(prefix)
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
        def _count_occ_before(text_str: str, token: str, end_pos: int) -> int:
            count = 0
            cursor = 0
            while True:
                pos = text_str.find(token, cursor)
                if pos < 0 or pos >= end_pos:
                    break
                count += 1
                cursor = pos + len(token)
            return count

        expanded_text = []
        for idx, t in enumerate(text):
            if DEFAULT_IMAGE_TOKEN in t:
                num_images = 1
                if num_images_per_sample is not None and idx < len(num_images_per_sample):
                    num_images = num_images_per_sample[idx]
                replacement = self.image_token_joiner.join([DEFAULT_IMAGE_TOKEN] * (self.image_seq_length * num_images))
                if span_list is not None:
                    span_start, span_end = span_list[idx]
                    delta = len(replacement) - len(DEFAULT_IMAGE_TOKEN)
                    if delta != 0 and span_start > 0:
                        shift = _count_occ_before(t, DEFAULT_IMAGE_TOKEN, span_start) * delta
                        span_list[idx][0] = span_start + shift
                        span_list[idx][1] = span_end + shift
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

        use_offsets = bool(span_list is not None and getattr(self.tokenizer, "is_fast", False))
        model_inputs = self.tokenizer(
            final_text,
            return_tensors=return_tensors,
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", None),
            return_offsets_mapping=use_offsets,
        )

        if span_list is not None and not use_offsets:
            logger.warning(
                "Tokenizer is not fast tokenizer; skip instruction_token_mask building."
            )

        if span_list is not None and use_offsets:
            offsets = model_inputs.get("offset_mapping", None)
            input_ids = model_inputs.get("input_ids", None)
            attention_mask = model_inputs.get("attention_mask", None)
            if (
                isinstance(offsets, torch.Tensor)
                and isinstance(input_ids, torch.Tensor)
                and offsets.ndim == 3
                and offsets.shape[:2] == input_ids.shape[:2]
            ):
                instruction_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for bid, (span_start, span_end) in enumerate(span_list):
                    if span_end <= span_start:
                        continue
                    tok_start = offsets[bid, :, 0]
                    tok_end = offsets[bid, :, 1]
                    valid = tok_end > tok_start
                    in_span = (tok_start < span_end) & (tok_end > span_start) & valid
                    if isinstance(attention_mask, torch.Tensor):
                        in_span = in_span & attention_mask[bid].to(dtype=torch.bool)
                    instruction_token_mask[bid] = in_span
                model_inputs["instruction_token_mask"] = instruction_token_mask

                if _is_rank0() and self._cot_mask_log_counter < self._cot_mask_log_first_n:
                    max_show = min(int(input_ids.shape[0]), 2)
                    for bid in range(max_show):
                        ids = input_ids[bid]
                        if isinstance(attention_mask, torch.Tensor):
                            active = attention_mask[bid].to(dtype=torch.bool)
                        else:
                            active = torch.ones_like(ids, dtype=torch.bool)
                        lang_active = active & (ids != int(self.image_token_id))
                        instr_sel = lang_active & instruction_token_mask[bid]
                        tmpl_sel = lang_active & (~instruction_token_mask[bid])
                        instr_ids = ids[instr_sel]
                        tmpl_ids = ids[tmpl_sel]
                        instr_text = self.tokenizer.decode(instr_ids.tolist(), skip_special_tokens=False)
                        tmpl_text = self.tokenizer.decode(tmpl_ids.tolist(), skip_special_tokens=False)

                        def _short_text(text_value: str, max_len: int = 220) -> str:
                            text_value = text_value.replace("\n", " ").strip()
                            if len(text_value) > max_len:
                                return text_value[:max_len] + "..."
                            return text_value

                        logger.info(
                            "[cot_query_mask] sample=%d lang=%d instr=%d tmpl=%d instr_ids_head=%s tmpl_ids_head=%s instr_text=%s tmpl_text=%s",
                            bid,
                            int(lang_active.sum().item()),
                            int(instr_sel.sum().item()),
                            int(tmpl_sel.sum().item()),
                            instr_ids[:16].tolist(),
                            tmpl_ids[:16].tolist(),
                            _short_text(instr_text),
                            _short_text(tmpl_text),
                        )
                    self._cot_mask_log_counter += 1

            model_inputs.pop("offset_mapping", None)
        
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
