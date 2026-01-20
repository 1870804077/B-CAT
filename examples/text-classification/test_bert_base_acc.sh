<<<<<<< HEAD
#!/bin/bash

# =======================================================
# 1. 网络与环境配置
# =======================================================
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SHAFT_LIB_PATH="$SCRIPT_DIR/.."
export PYTHONPATH=$PYTHONPATH:.:$SHAFT_LIB_PATH

TASK="sst2"
BATCH_SIZE_TRAIN=32
BATCH_SIZE_INFER=1
OUTPUT_ROOT="./checkpoints"

# 本地模型目录
LOCAL_MODEL_DIR="./bert-base-uncased"
# 微调后产出的模型目录
APPROX_MODEL_DIR="$OUTPUT_ROOT/test_var_${TASK}_approx"

echo "======================================================="
echo "🚀 SHAFT Test: $TASK"
echo "   Device: ($CUDA_VISIBLE_DEVICES)"
echo "   Mirror: $HF_ENDPOINT"
echo "======================================================="

# =======================================================
# 2. 智能轻量化下载 (只下 400MB，不下载 4GB)
# =======================================================
echo -e "\n🛠️  Checking Base Model..."

if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "   Local model not found. Downloading PyTorch weights ONLY..."
    
    if ! command -v huggingface-cli &> /dev/null; then
        echo "⚠️  huggingface-cli not found. Installing..."
        pip install -U "huggingface_hub[cli]"
    fi

    # 【核心修改】使用 --include 只下载必要文件
    # 排除 Flax, TensorFlow, Rust 等无用大文件
    huggingface-cli download google-bert/bert-base-uncased \
        --local-dir "$LOCAL_MODEL_DIR" \
        --local-dir-use-symlinks False \
        --resume-download \
        --include "config.json" "pytorch_model.bin" "vocab.txt" "tokenizer.json" "tokenizer_config.json" "*.safetensors"

    if [ $? -ne 0 ]; then
        echo "❌ Download failed. Please check network."
        exit 1
    fi
else
    echo "✅ Base model found at: $LOCAL_MODEL_DIR"
fi

# =======================================================
# Step 1: 明文微调
# =======================================================
echo -e "\n[Step 1/3] Starting Plaintext Fine-tuning..."

# 传入本地精简版模型路径
python model_modify.py \
    --model_name_or_path "$LOCAL_MODEL_DIR" \
    --task $TASK \
    --output_dir $OUTPUT_ROOT \
    --batch_size $BATCH_SIZE_TRAIN \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --approx_num_train_epochs 1 \
    --approx_learning_rate 1e-5 

if [ $? -ne 0 ]; then
    echo "❌ Training failed! Exiting."
    exit 1
fi

# =======================================================
# Step 2: 明文评估
# =======================================================
echo -e "\n[Step 2/3] Running Plaintext Evaluation..."

python run_glue_eval.py \
    --model_name_or_path "$APPROX_MODEL_DIR" \
    --task_name $TASK \
    --per_device_eval_batch_size 32 \
    --output_dir "${APPROX_MODEL_DIR}/eval_plain"

# =======================================================
# Step 3: 密文推理
# =======================================================
echo -e "\n[Step 3/3] Running SHAFT Encrypted Inference..."

python run_glue_private.py \
    --model_name_or_path "$APPROX_MODEL_DIR" \
    --task_name $TASK \
    --acc \
    --max_length 128 \
    --per_device_eval_batch_size $BATCH_SIZE_INFER \
    --output_dir "${APPROX_MODEL_DIR}/eval_private" \
    --ignore_mismatched_sizes

echo "======================================================="
echo "✅ All Done! Results saved in $APPROX_MODEL_DIR"
echo "======================================================="
=======
export TASK_NAME=sst2

python run_glue_private.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --max_length 128 \
  --acc \
  --per_device_eval_batch_size 1 \
  --output_dir eval_private/$TASK_NAME/
>>>>>>> main
