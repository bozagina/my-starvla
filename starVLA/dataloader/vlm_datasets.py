import os
import sys
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers
import tokenizers
from omegaconf import OmegaConf
from starVLA.dataloader.qwenvl_llavajson.qwen_data_config import data_list
from starVLA.dataloader.qwenvl_llavajson.rope2d import get_rope_index_25, get_rope_index_2

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from LLaVA_3D.llava import conversation as conversation_lib
from LLaVA_3D.llava.mm_utils import tokenizer_special_token

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>\n"
DEFAULT_VIDEO_TOKEN = "<video>\n"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mapanything_visual(
    sources,
    processor,
) -> Dict:
    if not isinstance(sources, (list, tuple)) or len(sources) == 0:
        raise ValueError("sources must be a non-empty list for mapanything_llava3d.")
    if processor is None:
        raise ValueError("processor must not be None for mapanything_llava3d.")
    sources_copy = copy.deepcopy(sources)
    sources_mm = preprocess_multimodal_llava3d(sources_copy)
    out = preprocess_v1_llava3d(sources_mm, processor.tokenizer, has_image=True)
    if not isinstance(out, dict):
        raise TypeError("LLaVA3D preprocess must return a dict for mapanything_llava3d.")
    if "input_ids" not in out or "labels" not in out:
        raise KeyError("LLaVA3D preprocess output must contain 'input_ids' and 'labels' for mapanything_llava3d.")
    input_ids = out["input_ids"]
    labels = out["labels"]
    if not torch.is_tensor(input_ids) or not torch.is_tensor(labels):
        raise TypeError("input_ids and labels must be torch.Tensor for mapanything_llava3d.")
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must be 2D tensors [B, T] for mapanything_llava3d.")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape for mapanything_llava3d.")
    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess_multimodal_llava3d(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence.get("value", "") or DEFAULT_VIDEO_TOKEN in sentence.get("value", ""):
                v = sentence["value"]
                v = v.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
                v = v.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                v = DEFAULT_IMAGE_TOKEN + "\n" + v
                v = v.strip()
                sentence["value"] = v
    return sources


IS_TOKENIZER_GREATER_THAN_0_14 = tokenizers.__version__ >= "0.14"


def preprocess_v1_llava3d(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    conv = conversation_lib.conv_templates.get("llava_v1", conversation_lib.default_conversation).copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_special_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_special_token(rou, tokenizer))
                instruction_len = len(tokenizer_special_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["data_path"] != "":
                    ann["data_path"] = data["data_path"]
                elif "raw_data" in ann.keys():
                    ann["data_path"] = ann["raw_data"]["data_root"]
            list_data_dict += annotations

        list_data_dict = self.pre_filter_long_case(list_data_dict, max_words=tokenizer.max_len_single_sentence)
        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.processor = getattr(data_args, "processor", None)

        rank0_print(f"Total training samples: {len(self.list_data_dict)}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    def pre_filter_long_case(self, list_data_dict, max_words=1024):
        """filter out conversations with total words exceeding max_words"""

        def count_total_words(convs):
            total = 0
            for entry in convs:
                value = entry.get("value", "")
                total += len(value.strip().split())
            return total

        return [item for item in list_data_dict if count_total_words(item.get("conversations", [])) <= max_words]

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("images" in sample) or ("videos" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        # if fix image size?
        if getattr(self.data_args, "fix_image_size", None) is not None:
            image = image.resize(
                self.data_args.fix_image_size,
                resample=Image.BICUBIC,
            )
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        if "image_grid_thw" in visual_processed:
            grid_thw = visual_processed["image_grid_thw"][0]
        else:
            grid_thw = torch.tensor([1, 1, 1])
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        if "images" in sources[0] and len(sources[0]["images"]):
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["images"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            if self.model_type == "mapanything_llava3d":
                sources_conv = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_mapanything_visual(sources_conv, self.processor)
                input_ids = data_dict["input_ids"]
                labels = data_dict["labels"]
                position_ids = torch.arange(0, input_ids.size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)
            else:
                grid_thw_merged = copy.deepcopy(grid_thw)
                if not isinstance(grid_thw, Sequence):
                    grid_thw_merged = [grid_thw_merged]
                    grid_thw = [grid_thw]
                grid_thw_merged = [
                    merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
                ]
                sources_conv = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_qwen_2_visual(
                    sources_conv, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
                )
                input_ids = data_dict["input_ids"]
                labels = data_dict["labels"]
                position_ids, _ = self.get_rope_index(
                    self.data_args.image_processor.merge_size,
                    input_ids,
                    torch.stack(grid_thw, dim=0),
                )
        elif "videos" in sources[0] and len(sources[0]["videos"]):
            video_file = self.list_data_dict[i]["videos"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources_conv = copy.deepcopy([e["conversations"] for e in sources])
            if self.model_type == "mapanything_llava3d":
                data_dict = preprocess_mapanything_visual(sources_conv, self.processor)
                input_ids = data_dict["input_ids"]
                labels = data_dict["labels"]
                position_ids = torch.arange(0, input_ids.size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)
            else:
                data_dict = preprocess_qwen_2_visual(sources_conv, self.tokenizer, grid_thw=grid_thw_merged)
                input_ids = data_dict["input_ids"]
                labels = data_dict["labels"]
                position_ids = torch.arange(0, input_ids.size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=input_ids[0],
                labels=labels[0],
                position_ids=position_ids,
            )
        if "images" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "videos" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        max_len = self.tokenizer.max_len_single_sentence
        if data_dict["input_ids"].shape[0] > max_len:
            data_dict["input_ids"] = data_dict["input_ids"][:max_len]
            data_dict["labels"] = data_dict["labels"][:max_len]
            data_dict["position_ids"] = position_ids[:, :, :max_len]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side=self.tokenizer.padding_side
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[..., : self.tokenizer.model_max_length]  # 3,bs,length

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor([0] + [len(seq) for seq in input_ids], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # load training dataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    # load evaluation dataset (if specified eval dataset path)
    eval_dataset = None
    if hasattr(data_args, "eval_dataset") and data_args.eval_dataset:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset_use = data_args.eval_dataset
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=eval_data_args)

    # select appropriate collator based on whether data needs to be flattened
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_vlm_dataloader(cfg):
    data_args = cfg.datasets.vlm_data

    if getattr(data_args, "model_type", None) == "mapanything_llava3d":
        import os
        import json
        from transformers import AutoImageProcessor, AutoTokenizer
        from starVLA.mapanything_llava3d.model.processing_mapanything_llava3d import (
            MapAnythingLlava3DProcessor,
        )

        base_dir = cfg.framework.mapanything_llava3d.base_vlm
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "r") as f:
            conf = json.load(f)

        vision_path = conf.get("vision_model_name_or_path")
        text_path = conf.get("language_model_name_or_path")

        image_processor = AutoImageProcessor.from_pretrained(
            vision_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_path,
            model_max_length=data_args.model_max_length,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )
        processor = MapAnythingLlava3DProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError("something wrong in make_vlm_dataloader.")
        # image_processor = AutoProcessor.from_pretrained(
        #     cfg.framework.qwenvlmapanything_llava3d.base_vlm,
        # ).image_processor

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.framework.mapanything_llava3d.base_vlm,
            model_max_length=data_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )

    # avoid processing these in dataset
    image_processor.max_pixels = int(data_args.max_pixels)
    image_processor.min_pixels = int(data_args.min_pixels)
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor
    if getattr(data_args_ns, "model_type", None) == "mapanything_llava3d":
        data_args_ns.processor = processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=4,
    )

    return {
        "train_dataloader": train_dataloader,
    }


from transformers import AutoTokenizer, AutoProcessor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/CoTrainVLM/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    dl = make_vlm_dataloader(cfg)
    train_dataloader = dl["train_dataloader"]
    print("VLM dataloader built successfully.")
    print("Num batches:", len(train_dataloader))
