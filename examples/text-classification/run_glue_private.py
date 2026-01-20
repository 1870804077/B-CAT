# coding=utf-8
# coding=utf-8
# Modified by B-CAT's team & Adapted for Custom Operators
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing a Transformers model in priavte for sequence classification on GLUE."""


# coding=utf-8
# Modified by SHAFT's team & Adapted for Custom Operators & RTX 3090
"""Testing a Transformers model in private for sequence classification on GLUE."""

import argparse
import json
import logging
import os
import time
import sys

import datasets
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    BertForSequenceClassification
)

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import crypten as ct
from crypten.config import cfg
try:
    from multiprocess_launcher import MultiProcessLauncher
except ImportError:
    MultiProcessLauncher = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

check_min_version("4.42.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run private inference on GLUE")
    parser.add_argument("--task_name", type=str, default=None, choices=list(task_to_keys.keys()))
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--num_data", type=int, default=-1, help="Debug: limit validation samples")
    parser.add_argument("--len_data", type=int, default=-1)
    parser.add_argument("--comp", action="store_true", help="Estimate computation cost")
    parser.add_argument("--acc", action="store_true", help="Evaluate accuracy")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (keep small for MPC)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", type=bool, default=False)
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    args = parser.parse_args()

    if args.task_name is None and args.validation_file is None:
        raise ValueError("Need either a task name or a validation file.")
    return args

def main():
    script_start_time = time.time()
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    if args.task_name is not None:
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        data_files = {"validation": args.validation_file}
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"
    is_regression = args.task_name == "stsb"
    
    if not is_regression:
        label_list = raw_datasets[validation_key].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # 2. Load Configuration
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    # 尝试加载 tokenizer，如果 checkpoint 里没有则用默认
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    except:
        print("⚠️  Tokenizer not found in path, using bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=not args.use_slow_tokenizer)

    # 3. 模型加载逻辑
    print(f"🔥 SHAFT Mode: Loading weights from {args.model_name_or_path} ...")
    
    model = AutoModelForSequenceClassification.from_config(config)

    checkpoint_file = os.path.join(args.model_name_or_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_file):
        print(f"   Found checkpoint: {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Weights loaded. Missing: {len(missing)}, Unexpected (Approx Params): {len(unexpected)}")
    else:
        print("   ⚠️ Checkpoint not found! Using random init (Validation will be poor).")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # 4. Preprocess Data
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", None

    padding = "max_length"
    
    def preprocess_function(examples):
        # 【核心修复】：变量名改为 inputs，避免覆盖全局 args
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        # 使用全局 args.max_length
        result = tokenizer(*inputs, padding=padding, max_length=args.max_length, truncation=True)
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names, # 确保移除文本列
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets[validation_key]
    data_collator = default_data_collator
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    metric = evaluate.load("glue", args.task_name) if args.task_name else evaluate.load("accuracy")

    # 5. Crypten 加密
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ct.init()
    
    print("🔒 Encrypting model...")
    dummy_input_ids = torch.zeros(1, args.max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, args.max_length, dtype=torch.long)
    dummy_token_type = torch.zeros(1, args.max_length, dtype=torch.long)
    
    private_model = ct.nn.from_pytorch(model, (dummy_input_ids, dummy_mask, dummy_token_type)).encrypt().to(device)
    print("✅ Model encrypted!")
    
    del model
    torch.cuda.empty_cache()

    # 6. 推理循环
    print("🚀 Starting Private Inference Loop...")
    samples_seen = 0
    
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Inference")):
        if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
            continue

        inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
        attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
        token_type_enc = ct.cryptensor(batch["token_type_ids"]).to(device)

        with ct.no_grad():
            outputs_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)
            outputs = outputs_enc.get_plain_text().cpu()

        predictions = outputs.argmax(dim=-1) if not is_regression else outputs.squeeze()
        references = batch["labels"]

        metric.add_batch(predictions=predictions, references=references)
        samples_seen += references.shape[0]
        
        del inputs_enc, attention_mask_enc, token_type_enc, outputs_enc
        torch.cuda.empty_cache()

        if args.num_data > 0 and samples_seen >= args.num_data:
            break
    
    eval_metric = metric.compute()
    print(f"\n🎯 Final Metric: {eval_metric}")
    print(f"⏱️ Total Time: {time.time() - script_start_time:.2f}s")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(eval_metric, f)

if __name__ == "__main__":
    args = parse_args()
    if args.acc:
         with cfg.temp_override({"cost.estimate_cost": False}):
            main()
    else:
        main()