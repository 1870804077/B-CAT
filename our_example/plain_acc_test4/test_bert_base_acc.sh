#!/bin/bash

# # =======================================================
# # 1. 网络与环境配置
# # =======================================================
# export CUDA_VISIBLE_DEVICES=0
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_DOWNLOAD_TIMEOUT=120

# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# SHAFT_LIB_PATH="$SCRIPT_DIR/.."
# export PYTHONPATH=$PYTHONPATH:.:$SHAFT_LIB_PATH

# TASK="sst2"
# BATCH_SIZE_TRAIN=32
# # 将推理 Batch Size 设为 1 以避免 MPC 显存溢出 (OOM)
# BATCH_SIZE_INFER=1 
# OUTPUT_ROOT="./checkpoints"

# # 本地模型目录
# LOCAL_MODEL_DIR="./bert-base-uncased"
# # 微调后产出的模型目录
# APPROX_MODEL_DIR="$OUTPUT_ROOT/test_var_${TASK}_approx"

# # echo "======================================================="
# # echo "🚀 SHAFT Test: $TASK"
# # echo "   Device: ($CUDA_VISIBLE_DEVICES)"
# # echo "   Mirror: $HF_ENDPOINT"
# # echo "======================================================="

# # # =======================================================
# # # 2. 智能轻量化下载 (升级版：检查具体文件)
# # # =======================================================
# # echo -e "\n🛠️  Checking Base Model..."

# # # 检查关键文件是否存在，而不仅仅是文件夹
# # if [ ! -f "$LOCAL_MODEL_DIR/vocab.txt" ] || [ ! -f "$LOCAL_MODEL_DIR/pytorch_model.bin" ]; then
# #     echo "   ⚠️  Model files missing or incomplete. Downloading PyTorch weights..."
    
# #     if ! command -v huggingface-cli &> /dev/null; then
# #         echo "   Installing huggingface-cli..."
# #         pip install -U "huggingface_hub[cli]"
# #     fi

# #     # 只下载必要文件 (约 400MB)
# #     huggingface-cli download google-bert/bert-base-uncased \
# #         --local-dir "$LOCAL_MODEL_DIR" \
# #         --local-dir-use-symlinks False \
# #         --resume-download \
# #         --include "config.json" "pytorch_model.bin" "vocab.txt" "tokenizer.json" "tokenizer_config.json" "*.safetensors"

# #     if [ $? -ne 0 ]; then
# #         echo "❌ Download failed. Please check network."
# #         exit 1
# #     fi
# #     echo "✅ Download finished."
# # else
# #     echo "✅ Base model verified at: $LOCAL_MODEL_DIR"
# # fi

# # =======================================================
# # Step 1: 明文微调
# # =======================================================
# echo -e "\n[Step 1/3] Starting Plaintext Fine-tuning..."

# python model_modify.py \
#     --model_name_or_path "$LOCAL_MODEL_DIR" \
#     --task $TASK \
#     --output_dir $OUTPUT_ROOT \
#     --batch_size $BATCH_SIZE_TRAIN \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --approx_num_train_epochs 1 \
#     --approx_learning_rate 1e-5 

# if [ $? -ne 0 ]; then
#     echo "❌ Training failed! Exiting."
#     exit 1
# fi

# # # =======================================================
# # # Step 2: 明文评估
# # # =======================================================
# # echo -e "\n[Step 2/3] Running Plaintext Evaluation..."

# # python run_glue_eval.py \
# #     --model_name_or_path "$APPROX_MODEL_DIR" \
# #     --task_name $TASK \
# #     --per_device_eval_batch_size 32 \
# #     --output_dir "${APPROX_MODEL_DIR}/eval_plain"

# # =======================================================
# # Step 3: 密文推理
# # =======================================================
# # echo -e "\n[Step 3/3] Running SHAFT Encrypted Inference..."

# # python run_glue_private.py \
# #     --model_name_or_path "$APPROX_MODEL_DIR" \
# #     --task_name $TASK \
# #     --acc \
# #     --max_length 128 \
# #     --per_device_eval_batch_size $BATCH_SIZE_INFER \
# #     --output_dir "${APPROX_MODEL_DIR}/eval_private" \
# #     --ignore_mismatched_sizes

# echo "======================================================="
# echo "✅ All Done! Results saved in $APPROX_MODEL_DIR"
# echo "======================================================="



#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# 定义结果输出文件
RESULT_FILE="./results.txt"

# 初始化结果文件
echo "==========================================================" > "$RESULT_FILE"
echo "🚀 Batch Benchmark Started at $(date)" | tee -a "$RESULT_FILE"
echo "==========================================================" | tee -a "$RESULT_FILE"

run_task() {
    TASK_NAME=$1
    EPOCHS=$2

    echo "" | tee -a "$RESULT_FILE"
    echo "##########################################################" | tee -a "$RESULT_FILE"
    echo "▶️  STARTING TASK: $TASK_NAME (Epochs: $EPOCHS)" | tee -a "$RESULT_FILE"
    echo "##########################################################" | tee -a "$RESULT_FILE"

    # 【修改】加入了 --disable_tqdm 参数
    python model_modify.py \
        --task "$TASK_NAME" \
        --num_train_epochs "$EPOCHS" \
        --approx_num_train_epochs "$EPOCHS" \
        # --disable_tqdm \
        2>&1 | tee -a "$RESULT_FILE"
    
    echo "" | tee -a "$RESULT_FILE"
    echo "✅ Finished $TASK_NAME" | tee -a "$RESULT_FILE"
}

# ==========================================
# 任务列表
# ==========================================
run_task "sst2" 1
run_task "qnli" 1
run_task "cola" 3

echo "" | tee -a "$RESULT_FILE"
echo "==========================================================" | tee -a "$RESULT_FILE"
echo "🎉 All tasks completed successfully!" | tee -a "$RESULT_FILE"
echo "==========================================================" | tee -a "$RESULT_FILE"