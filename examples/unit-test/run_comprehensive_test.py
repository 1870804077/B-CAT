'''
goal：采用这个文件来测试全部算子在不同大小的输入的情况下的输出

input：test type && test input scale

output:

print(f"({gelu_size[0]}, {gelu_size[1]}) "
                f"time: {gelu_time[gelu_size[1]] / runs:.4f}s, "
                f"bytes: {gelu_bytes[gelu_size[1]] / 1048576 / runs:.0f} MB, "
                f"rounds: {gelu_rounds[gelu_size[1]] / runs:.0f}"
            )

'''

import time
import torch
import torch.nn.functional as F
import crypten
from multiprocess_launcher import MultiProcessLauncher

# =============================================================================
# 1. 真实模型尺寸定义 (Batch=1, Seq=128)
# =============================================================================

# FFN 层尺寸: (Seq_Len, Intermediate_Size)
FFN_BASE  = (128, 3072)  # BERT-Base
FFN_LARGE = (128, 4096)  # BERT-Large

# Attention Scores 尺寸: (Batch * Num_Heads * Seq_Len, Seq_Len)
# Base: 1 * 12 * 128 = 1536
# Large: 1 * 16 * 128 = 2048
ATTN_BASE  = (1536, 128)
ATTN_LARGE = (2048, 128)

# Normalization Scalars 尺寸 (Sum or Var): (Dim0, 1)
# 1/x 用于 Softmax: Dim0 = Batch * Heads * Seq
NORM_SOFTMAX_BASE  = (1536, 1)
NORM_SOFTMAX_LARGE = (2048, 1)
# 1/sqrtx 用于 LayerNorm: Dim0 = Batch * Seq (LayerNorm 对 hidden 维度归一化，得到每个token的标量var)
NORM_LN_BASE = (128, 1) 

# =============================================================================
# 2. 测试配置
# =============================================================================
TEST_CONFIGS = {
    # --- FFN Activation 类 (密集计算) ---
    "GELU": {
        "torch_fn": F.gelu,
        "crypten_fn": lambda x: x.gelu(approximate="none"),
        "inputs": [FFN_BASE, FFN_LARGE],
        "data_mode": "zeros"
    },
    "SiLU": {
        "torch_fn": F.silu,
        "crypten_fn": lambda x: x * x.sigmoid(), # SiLU = x * sigmoid(x)
        "inputs": [FFN_BASE, FFN_LARGE],
        "data_mode": "zeros"
    },
    "TANH": {
        "torch_fn": torch.tanh,
        "crypten_fn": "tanh",
        "inputs": [FFN_BASE, FFN_LARGE],
        "data_mode": "random" # Tanh 在 0 附近近似线性，用随机数测更准确
    },
    "SIN": {
        "torch_fn": torch.sin,
        "crypten_fn": "sin",
        "inputs": [FFN_BASE, FFN_LARGE],
        "data_mode": "random"
    },

    # --- Attention 类 (矩阵操作) ---
    "SOFTMAX": {
        "torch_fn": lambda x: F.softmax(x, dim=-1),
        "crypten_fn": lambda x: x.softmax(dim=-1),
        "inputs": [ATTN_BASE, ATTN_LARGE],
        "data_mode": "random"
    },
    "EXP": {
        "torch_fn": torch.exp,
        "crypten_fn": "exp",
        "inputs": [ATTN_BASE, ATTN_LARGE],
        "data_mode": "random"
    },

    # --- Normalization 类 (标量/向量操作) ---
    "RECIPROCAL (1/x)": {
        "torch_fn": torch.reciprocal,
        "crypten_fn": "reciprocal",
        "inputs": [NORM_SOFTMAX_BASE, NORM_SOFTMAX_LARGE], # 模拟 Softmax 归一化
        "data_mode": "positive" # 必须 > 0
    },
    "INV_SQRT (1/sqrtx)": {
        "torch_fn": torch.rsqrt,
        "crypten_fn": lambda x: x.inv_sqrt() if hasattr(x, 'inv_sqrt') else x.pow(-0.5),
        "inputs": [NORM_LN_BASE, NORM_SOFTMAX_LARGE], # 模拟 LN 和 大规模并行
        "data_mode": "positive" # 必须 > 0
    }
}

OPERATORS_TO_RUN = ["SIN", "TANH", "EXP", "RECIPROCAL (1/x)", "INV_SQRT (1/sqrtx)", "GELU", "SiLU", "SOFTMAX"]

# =============================================================================
# 3. 辅助函数 & 主逻辑
# =============================================================================
def get_data(shape, mode, device):
    if mode == "zeros":
        return crypten.cryptensor(torch.zeros(shape), device=device)
    elif mode == "positive":
        # [0.5, 2.5] 避免 0 和极端值
        return crypten.cryptensor(torch.rand(shape, device=device) * 2 + 0.5)
    else: # random
        return crypten.cryptensor(torch.randn(shape, device=device))

def run_benchmark():
    crypten.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = 10
    
    rank = crypten.comm.get().get_rank()

    if rank == 0:
        print(f"{'Operator':<20} | {'Input Size':<15} | {'Time(s)':<10} | {'Comm(MB)':<10} | {'Rounds':<10}")
        print("-" * 75)

    for op_name in OPERATORS_TO_RUN:
        cfg = TEST_CONFIGS[op_name]
        
        # 提取执行函数
        c_func = cfg["crypten_fn"]
        if isinstance(c_func, str):
            exec_func = lambda x: getattr(x, c_func)()
        else:
            exec_func = c_func

        for shape in cfg["inputs"]:
            # 1. 准备数据
            input_data = get_data(shape, cfg["data_mode"], device)
            size_str = str(shape)

            # 2. 预热 (Warmup) - 跑一次不计入统计
            # _ = exec_func(input_data)
            # crypten.reset_communication_stats() # 重置

            # 3. 循环测试
            crypten.reset_communication_stats()
            start = time.time()
            for _ in range(runs):
                res = exec_func(input_data)
                # 强制同步: 如果是 lazy evaluation 需要在这里获取结果
                # res.get_plain_text() 
                # 但 Crypten 通常是 eager 的，除了 comms
            
            # 4. 统计
            duration = time.time() - start
            stats = crypten.get_communication_stats()
            
            avg_time = duration / runs
            avg_mb = stats['bytes'] / 1048576 / runs
            avg_rounds = stats['rounds'] / runs

            if rank == 0:
                print(f"{op_name:<20} | {size_str:<15} | {avg_time:<10.4f} | {avg_mb:<10.0f} | {avg_rounds:<10.0f}")

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, run_benchmark)
    launcher.start()
    launcher.join()
    launcher.terminate()