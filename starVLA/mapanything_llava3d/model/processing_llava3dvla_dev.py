# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from typing import List, Optional, Union, Dict
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import Unpack, _validate_images_text_input_order, ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.models.paligemma.processing_paligemma import (
    make_batched_images, 
    _is_str_or_image, 
    PaliGemmaProcessorKwargs
)
# 使用LLaVA-3D的特殊标记替代PaliGemma的标记
from LLaVA_3D.llava.constants import (
    DEFAULT_IMAGE_TOKEN as IMAGE_TOKEN, # 正确
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX
)
# 定义额外的标记
EXTRA_TOKENS = [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER]
# 使用LLaVA-3D的特殊标记替代PaliGemma的标记
# ...
from action_tokenizer import SpatialActionTokenizer
logger = logging.get_logger(__name__)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`list[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    repeated = " ".join([image_token] * (image_seq_len * num_images))
    return f"{repeated}{bos_token}{prompt}"

import json
from pathlib import Path

TOKENIZER_CLASS = (
    "LlamaTokenizer", "LlamaTokenizerFast",
    "GemmaTokenizer", "GemmaTokenizerFast",
)


def _load_processor_config_fallback(tokenizer):
    """
    When running `AutoProcessor.from_pretrained`, HuggingFace only forwards `image_processor` and `tokenizer`
    instances to the processor constructor. Extra kwargs such as `statistics`, `bin_policy`, etc. are stored
    inside `processor_config.json`, so we need to reload them manually if they were not explicitly provided.
    """
    name_or_path = getattr(tokenizer, "name_or_path", None)
    candidate_dirs = []
    if name_or_path:
        candidate_dirs.append(Path(name_or_path))
        candidate_dirs.append(Path(name_or_path).parent)
    candidate_dirs.append(Path.cwd())
    for base in candidate_dirs:
        cfg_path = base / "processor_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    return json.load(f)
            except Exception as error:
                logger.warning(f"Failed to read processor_config.json from {cfg_path}: {error}")
                return {}
    return {}

class SpatialVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = TOKENIZER_CLASS

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
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
        # Derive image_seq_length when not explicitly provided on image_processor
        if not hasattr(image_processor, "image_seq_length"):
            cfg = _load_processor_config_fallback(tokenizer)
            seq_len = cfg.get("image_seq_length")
            if seq_len is None:
                # Heuristic: use patch14 if patch_size is not available on image_processor
                try:
                    h = image_processor.size["height"]
                    patch = getattr(image_processor, "patch_size", 14)
                    seq_len = int((h // patch) ** 2)
                except Exception:
                    seq_len = 256
            setattr(image_processor, "image_seq_length", int(seq_len))
        self.image_seq_length = int(getattr(image_processor, "image_seq_length"))

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id
            
        # 使用LLaVA-3D特殊标记，且在返回中携带索引以便模型转换
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.ignore_index = IGNORE_INDEX
        
        added_vocab = tokenizer.get_added_vocab()
        tokens_to_add_now = [t for t in EXTRA_TOKENS if t not in added_vocab]
        if tokens_to_add_now:
            tokenizer.add_tokens(tokens_to_add_now)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        # Ensure pad token exists for padding
        if getattr(tokenizer, "pad_token", None) is None or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token", None) else getattr(tokenizer, "bos_token", "<pad>")
        try:
            tokenizer.padding_side = "right"
        except Exception:
            pass

        # 如果提供了chat_template参数，优先使用提供的模板
        if chat_template is not None:
            tokenizer.chat_template = chat_template
            
        super().__init__(image_processor, tokenizer, chat_template=tokenizer.chat_template)

        # action tokenizer
        config_cache = None
        def _resolve_extra(field_name, current_value):
            nonlocal config_cache
            if current_value is not None:
                return current_value
            if config_cache is None:
                config_cache = _load_processor_config_fallback(tokenizer)
            return config_cache.get(field_name)

        statistics = _resolve_extra("statistics", statistics)
        bin_policy = _resolve_extra("bin_policy", bin_policy)
        intrinsic_config = _resolve_extra("intrinsic_config", intrinsic_config)
        action_config = _resolve_extra("action_config", action_config)
        missing = []
        if statistics is None:
            missing.append("statistics")
        if bin_policy is None:
            missing.append("bin_policy")
        if intrinsic_config is None:
            missing.append("intrinsic_config")
        if action_config is None:
            missing.append("action_config")
        if missing:
            raise ValueError(f"Missing required processor config fields: {', '.join(missing)}. "
                             "Ensure processor_config.json includes these entries or pass them explicitly.")

        self.statistics = statistics
        self.bin_policy = bin_policy
        self.min_sigma = min_sigma
        self.intrinsic_config = intrinsic_config
        self.action_config = action_config
        self.num_obs_steps = num_obs_steps
        self.obs_delta = obs_delta
        self.action_chunk_size = action_chunk_size
        self.dataset_intrinsics = {}
        height, width = image_processor.size["height"], image_processor.size["width"]

        # scale intrinsic matrix
        for k, v in intrinsic_config.items():
            K = torch.tensor(v["intrinsic"]).float()
            K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
            self.dataset_intrinsics[k] = K
        
        self.action_tokenizer = SpatialActionTokenizer(
            tokenizer=tokenizer, num_bins=action_config["num_bins"], 
            bin_policy=bin_policy, use_spherical=action_config["use_spherical"],
            min_sigma=min_sigma,
        )

        # Load optional fields from processor_config.json if present
        try:
            cfg = _load_processor_config_fallback(tokenizer)
            if isinstance(cfg, dict):
                self.action_chunk_size = int(cfg.get("action_chunk_size", self.action_chunk_size))
                self.num_obs_steps = int(cfg.get("num_obs_steps", self.num_obs_steps))
                self.obs_delta = int(cfg.get("obs_delta", self.obs_delta))
        except Exception:
            pass

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        unnorm_key: Optional[str] = None,
        suffix_actions: Optional[np.array] = None, # (t e)
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if suffix_actions is not None:
            action_tokens = self.action_tokenizer(suffix_actions) # (n,3)
            suffix="".join(action_tokens.flatten())
        else:
            suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once( "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model.")
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")
                if suffix is not None and _is_str_or_image(suffix): suffix = [suffix]
                if suffix is not None: suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
                # The output from build_string_from_input will be:
                # "<im><im><im><s>Initial str"
                # Ensure symmetric BOS/EOS and image token expansion for inputs without explicit <image>
                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    + self.tokenizer.eos_token
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_patch = " ".join([IMAGE_TOKEN] * self.image_seq_length)
                    expanded_sample = sample.replace(IMAGE_TOKEN, expanded_patch)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                # Append EOS to ensure symmetry with non-<image> branch; no trailing newline
                input_strings = [f"{sample}{self.tokenizer.eos_token}" for sample in expanded_samples]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        # Ensure batched tensors have consistent length: enable padding/truncation and set max_length
        tk_kwargs = output_kwargs["text_kwargs"]
        if tk_kwargs.get("max_length", None) is None:
            model_max_len = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(model_max_len, int) and model_max_len > 0:
                tk_kwargs["max_length"] = model_max_len
        if tk_kwargs.get("max_length", None) is not None:
            tk_kwargs["max_length"] = int(tk_kwargs["max_length"]) + int(self.image_seq_length)
        tk_kwargs["padding"] = tk_kwargs.get("padding", "longest")
        tk_kwargs.setdefault("truncation", True)
        # Ensure tensor return type is respected even if not merged into text_kwargs
        if kwargs.get("return_tensors", None) is not None and tk_kwargs.get("return_tensors", None) is None:
            tk_kwargs["return_tensors"] = kwargs["return_tensors"]

        # As a fallback, when fast tokenizer still returns ragged lists, enforce padding here
        # Avoid returning tensors directly to allow manual padding
        tk_kwargs = output_kwargs["text_kwargs"]
        tensor_type = tk_kwargs.pop("return_tensors", None)
        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **tk_kwargs,
        )
        if isinstance(inputs.get("input_ids"), list) or tensor_type is not None:
            max_len = max(len(seq) for seq in inputs["input_ids"]) if len(inputs["input_ids"]) > 0 else 0
            pad_id = self.tokenizer.pad_token_id
            padded_ids = [seq + [pad_id] * (max_len - len(seq)) for seq in inputs["input_ids"]]
            padded_mask = [mask + [0] * (max_len - len(mask)) for mask in inputs.get("attention_mask", [[1]*len(seq) for seq in inputs["input_ids"]])]
            inputs["input_ids"] = torch.tensor(padded_ids, dtype=torch.long)
            inputs["attention_mask"] = torch.tensor(padded_mask, dtype=torch.long)
            if return_token_type_ids and isinstance(inputs.get("token_type_ids"), list):
                padded_tti = [tti + [0] * (max_len - len(tti)) for tti in inputs["token_type_ids"]]
                inputs["token_type_ids"] = torch.tensor(padded_tti, dtype=torch.long)

        key = unnorm_key if (unnorm_key in self.dataset_intrinsics) else "default"
        intrinsic = self.dataset_intrinsics[key]
        # Attach both tokenizer real id and LLaVA sentinel index for downstream mapping
        return_data = {
            **inputs,
            "pixel_values": pixel_values,
            "intrinsic": intrinsic,
            "image_token_id": self.image_token_id,
            "image_token_index": self.image_token_index,
            "num_image_tokens": int(self.image_seq_length),
        }

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def decode_actions(
        self,
        generation_outputs: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        action_token_num = 3  # translation + rotation + gripper
        predicted_action_token_ids = generation_outputs[0, : action_token_num * self.action_chunk_size].detach().cpu().long().numpy()
        assert self.tokenizer.eos_token != predicted_action_token_ids[-1], "[error] actions contain EOS token, please check you truncation settings!"

        if predicted_action_token_ids.shape[0] < action_token_num * self.action_chunk_size:  # pad with zeros
            logger.warning(f"Padding zero action!")
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

        # Align stats length with decoded action dimension
        decoded_dim = normalized_action_chunks.shape[1]
        action_dim = len(action_norm_stats["q01"]) if isinstance(action_norm_stats.get("q01"), (list, np.ndarray)) else decoded_dim
        mask_cfg = action_norm_stats.get("mask", np.ones(action_dim))
        mask = np.array(mask_cfg, dtype=bool)
        action_high = np.array(action_norm_stats.get("q99", [1.0] * action_dim))
        action_low = np.array(action_norm_stats.get("q01", [-1.0] * action_dim))
        # If provided stats length mismatches decoded_dim, pad or trim to match
        if action_high.shape[0] != decoded_dim or action_low.shape[0] != decoded_dim or mask.shape[0] != decoded_dim:
            default_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
            default_high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0])
            default_mask = np.ones_like(default_high, dtype=bool)
            # build arrays of length decoded_dim
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
            # Note: last dimension assumed gripper; defaults set to [0,1]

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
