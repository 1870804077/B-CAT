import os
import sys
import argparse
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BertConfig
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np

# 添加当前路径以导入近似模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from approximation_plain import replace_bert_modules


def compute_metrics(eval_pred, task_name):
    """修正metric计算逻辑，适配所有GLUE任务"""
    metric = evaluate.load("glue", task_name)
    predictions, labels = eval_pred
    if task_name == "stsb":
        predictions = predictions[:, 0]
    elif task_name in ["cola", "sst2", "mrpc", "qnli", "rte"]:
        predictions = np.argmax(predictions, axis=1)
    else:
        raise ValueError(f"Unsupported task for metrics: {task_name}")
    result = metric.compute(predictions=predictions, references=labels)
    if task_name == "cola":
        return {"matthews_correlation": result["matthews_correlation"]}
    elif task_name == "mrpc":
        return {"accuracy": result["accuracy"], "f1": result["f1"]}
    else:
        return {"accuracy": result["accuracy"]}


def evaluate_model(model, val_dataset, tokenizer, task, batch_size, device):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            if task == "stsb":
                preds = logits.squeeze(-1)
            else:
                preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metric = evaluate.load("glue", task)
    if task == "stsb":
        result = metric.compute(predictions=all_preds, references=all_labels)["pearson"]
    elif task == "mrpc":
        acc = metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
        f1 = metric.compute(predictions=all_preds, references=all_labels)["f1"]
        result = {"accuracy": acc, "f1": f1}
    elif task == "cola":
        result = metric.compute(predictions=all_preds, references=all_labels)["matthews_correlation"]
    else:
        result = metric.compute(predictions=all_preds, references=all_labels)["accuracy"]

    return result, all_preds, all_labels


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sst2", 
                        choices=["sst2", "mrpc", "qnli", "rte", "cola", "stsb"])
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--approx_num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--approx_learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--skip_finetune", action="store_true", 
                        help="Skip fine-tuning even if model missing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积，适配小显存GPU")
    args = parser.parse_args()

    task = args.task.lower()
    orig_output_dir = os.path.join(args.output_dir, f"test_var_{task}")
    approx_output_dir = os.path.join(args.output_dir, f"test_var_{task}_approx")

    orig_model_path = os.path.join(orig_output_dir, "pytorch_model.bin")
    approx_model_path = os.path.join(approx_output_dir, "pytorch_model.bin")
    need_orig_finetune = not os.path.exists(orig_model_path)
    need_approx_finetune = not os.path.exists(approx_model_path)

    os.makedirs(orig_output_dir, exist_ok=True)
    os.makedirs(approx_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running SHAFT-style evaluation on GLUE/{task.upper()} (Device: {device})")
    print(f"📂 Loading Base Model from: {args.model_name_or_path}")

    # ==============================
    # Step 0: 数据预处理
    # ==============================
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path, 
        clean_up_tokenization_spaces=True,
        padding_side="right"
    )

    TASK_TO_KEYS = {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
    }

    if task not in TASK_TO_KEYS:
        raise ValueError(f"Task '{task}' not supported.")
    text_column_1, text_column_2 = TASK_TO_KEYS[task]

    def preprocess_function(examples):
        if text_column_2 is None:
            return tokenizer(examples[text_column_1], truncation=True, max_length=args.max_seq_length, padding=False)
        else:
            return tokenizer(examples[text_column_1], examples[text_column_2], truncation=True, max_length=args.max_seq_length, padding=False)

    train_dataset = load_dataset("glue", task, split="train")
    val_dataset = load_dataset("glue", task, split="validation")

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    # 显式赋值以修复 rename_column 问题
    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("label", "labels")
    if "label" in val_dataset.column_names:
        val_dataset = val_dataset.rename_column("label", "labels")
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_labels = 1 if task == "stsb" else 2
    metric_for_best_model = "eval_matthews_correlation" if task == "cola" else "eval_accuracy"
    if task == "stsb": metric_for_best_model = "eval_pearson"

    # ==============================
    # Step 1: 原生模型训练
    # ==============================
    model_orig = None
    if need_orig_finetune and not args.skip_finetune:
        print(f"🔍 Original model not found. Starting fine-tuning...")
        model_orig = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        training_args = TrainingArguments(
            output_dir=orig_output_dir,
            eval_strategy="epoch", save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            save_safetensors=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
        )
        # 【修改】使用 tokenizer=tokenizer 以获得更好的兼容性
        trainer = CustomTrainer(model=model_orig, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator, tokenizer=tokenizer, compute_metrics=lambda p: compute_metrics(p, task))
        trainer.train()
        trainer.save_model(orig_output_dir)
        # 【新增】显式保存 Tokenizer
        tokenizer.save_pretrained(orig_output_dir)
    else:
        print(f"✅ Found existing original model.")
        model_orig = BertForSequenceClassification.from_pretrained(orig_output_dir, num_labels=num_labels, ignore_mismatched_sizes=True)

    # ==============================
    # Step 2: 近似模型训练
    # ==============================
    model_approx = None
    if need_approx_finetune and not args.skip_finetune:
        print(f"🔍 Approximate model not found. Building and fine-tuning...")
        model_source = BertForSequenceClassification.from_pretrained(orig_output_dir, num_labels=num_labels, ignore_mismatched_sizes=True)
        model_approx = BertForSequenceClassification.from_pretrained(orig_output_dir, num_labels=num_labels, ignore_mismatched_sizes=True)
        
        print("   Replacing BERT modules...")
        model_approx.bert = replace_bert_modules(model_approx.bert)
        
        print("   🔥🔥 Transferring weights...")
        model_approx.load_state_dict(model_source.state_dict(), strict=False)
        del model_source
        torch.cuda.empty_cache()

        approx_training_args = TrainingArguments(
            output_dir=approx_output_dir,
            eval_strategy="epoch", save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.approx_learning_rate,
            num_train_epochs=args.approx_num_train_epochs,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            save_safetensors=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=torch.cuda.is_available(),
        )
        trainer = CustomTrainer(model=model_approx, args=approx_training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator, tokenizer=tokenizer, compute_metrics=lambda p: compute_metrics(p, task))
        trainer.train()
        trainer.save_model(approx_output_dir)
        # 【新增】显式保存 Tokenizer
        tokenizer.save_pretrained(approx_output_dir)
        print(f"✅ Approximate model AND tokenizer saved to {approx_output_dir}")
    else:
        print(f"✅ Found existing approximate model. Loading with STRUCTURE FIX...")
        config = BertConfig.from_pretrained(approx_output_dir, num_labels=num_labels)
        model_approx = BertForSequenceClassification(config)
        model_approx.bert = replace_bert_modules(model_approx.bert)
        state_dict = torch.load(approx_model_path, map_location="cpu")
        model_approx.load_state_dict(state_dict, strict=False)

    # Eval and Report
    if model_orig:
        print("📊 Evaluating Original...")
        model_orig.to(device)
        res, _, _ = evaluate_model(model_orig, val_dataset, tokenizer, task, args.batch_size, device)
        print(f"Original Score: {res}")
    
    if model_approx:
        print("📊 Evaluating Approx...")
        model_approx.to(device)
        res, _, _ = evaluate_model(model_approx, val_dataset, tokenizer, task, args.batch_size, device)
        print(f"Approx Score: {res}")

if __name__ == "__main__":
    main()