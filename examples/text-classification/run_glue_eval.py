# coding=utf-8
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
"""Testing a 🤗 Transformers model for sequence classification on GLUE."""

# coding=utf-8
# Modified for SHAFT Plaintext Simulation
"""Testing a 🤗 Transformers model for sequence classification on GLUE."""

import argparse
import logging
import os
import sys
import torch
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    BertForSequenceClassification
)

# 禁用 TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 导入近似算子
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from approximation_plain import replace_bert_modules
    print("✅ Successfully imported 'replace_bert_modules'")
except ImportError:
    print("⚠️  Warning: 'approximation_plain.py' not found in current directory.")
    print("    Simulated approximation will NOT be applied!")

task_to_keys = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=list(task_to_keys.keys()))
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Data
    raw_datasets = load_dataset("glue", args.task_name)
    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"
    is_regression = args.task_name == "stsb"
    
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets[validation_key].features["label"].names
        num_labels = len(label_list)

    # 2. Load Model & Apply Approx
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    # 注意：这里可能加载失败如果 tokenizer 没有保存，但 Step 1 修复后应该有了
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except:
        print("⚠️ Tokenizer not found in checkpoint, using default bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print(f"🔥 Plaintext Sim: Loading weights from {args.model_name_or_path} ...")
    model = BertForSequenceClassification(config)
    
    # 【核心】应用近似算子
    if 'replace_bert_modules' in globals():
        print("   Applying replace_bert_modules...")
        model.bert = replace_bert_modules(model.bert)
    
    # 加载权重
    checkpoint = os.path.join(args.model_name_or_path, "pytorch_model.bin")
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("   ✅ Weights loaded (strict=False).")
    
    model.to("cuda")
    model.eval()

    # 3. Process Data
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    
    def preprocess(examples):
        args_tok = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args_tok, padding="max_length", max_length=args.max_length, truncation=True)
        # 显式保留 label
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    # 【关键修复】remove_columns 确保只保留 tensor 友好的列
    eval_dataset = raw_datasets[validation_key].map(
        preprocess, 
        batched=True,
        remove_columns=raw_datasets[validation_key].column_names
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    metric = evaluate.load("glue", args.task_name)

    # 4. Inference
    print("🚀 Running Inference...")
    for batch in tqdm(dataloader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(predictions=preds, references=batch["labels"])

    print(f"🎯 Result: {metric.compute()}")

if __name__ == "__main__":
    main()