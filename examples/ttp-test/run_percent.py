import time
import torch
import crypten
import pandas as pd
import torch.multiprocessing as mp

# ==========================================
# 辅助函数：通用性能测试
# ==========================================
def run_benchmark(name, func, args, device, runs=5):
    """
    运行指定算子/层，统计时间、通信量和轮次
    """
    # 1. 预热 (Warm-up) - 建立通信连接，避免首轮握手延迟干扰
    for _ in range(2):
        func(*args)
    
    # 2. 重置统计器
    crypten.reset_communication_stats()
    
    # 3. 同步并计时
    if device == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(runs):
        func(*args)
        
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    # 4. 获取统计数据
    stats = crypten.get_communication_stats()
    
    return {
        "Component": name,
        "Time(ms)": (end_time - start_time) * 1000 / runs, # 毫秒
        "Comm(MB)": stats['bytes'] / (1024**2) / runs,     # MB
        "Rounds": stats['rounds'] / runs
    }

# ==========================================
# Worker 函数：子进程执行体
# ==========================================
def worker_main(device='cuda'):
    # 【核心修复】：在子进程入口强制开启 TTP 模式
    # 必须在任何 Crypten 张量创建之前设置！
    crypten.init()
    crypten.cfg.mpc.provider = "TTP"
    
    rank = crypten.comm.get().get_rank()
    if rank == 0:
        print(f"Worker {rank} started. Provider: {crypten.cfg.mpc.provider}, Device: {device}")

    # --- BERT-Base 配置参数 ---
    # Batch=1, SeqLen=128, Hidden=768
    # FFN Intermediate=3072, Heads=12
    BATCH = 1
    SEQ_LEN = 128
    HIDDEN = 768
    FFN_DIM = 3072
    HEADS = 12
    
    results = []

    # ---------------------------------------------------------
    # 1. Linear: Hidden -> Hidden (用于 Attention Q/K/V/Output)
    # ---------------------------------------------------------
    lin_h_h = crypten.nn.Linear(HIDDEN, HIDDEN).to(device).encrypt()
    x_h = crypten.cryptensor(torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device))
    results.append(run_benchmark("Linear (768->768)", lin_h_h, [x_h], device))

    # ---------------------------------------------------------
    # 2. Linear: Hidden -> FFN (用于 FFN Expansion)
    # ---------------------------------------------------------
    lin_h_f = crypten.nn.Linear(HIDDEN, FFN_DIM).to(device).encrypt()
    results.append(run_benchmark("Linear (768->3072)", lin_h_f, [x_h], device))

    # ---------------------------------------------------------
    # 3. Linear: FFN -> Hidden (用于 FFN Contraction)
    # ---------------------------------------------------------
    lin_f_h = crypten.nn.Linear(FFN_DIM, HIDDEN).to(device).encrypt()
    x_f = crypten.cryptensor(torch.randn(BATCH, SEQ_LEN, FFN_DIM, device=device))
    results.append(run_benchmark("Linear (3072->768)", lin_f_h, [x_f], device))

    # ---------------------------------------------------------
    # 4. GELU (用于 FFN 激活)
    # ---------------------------------------------------------
    # 输入维度是 FFN_DIM
    results.append(run_benchmark("GELU (3072)", x_f.gelu, [], device))

    # ---------------------------------------------------------
    # 5. Softmax (用于 Attention Score)
    # ---------------------------------------------------------
    # Attention Score 维度: (Batch, Heads, SeqLen, SeqLen)
    x_attn = crypten.cryptensor(torch.randn(BATCH, HEADS, SEQ_LEN, SEQ_LEN, device=device))
    results.append(run_benchmark("Softmax (Attn)", x_attn.softmax, [-1], device))

    # ---------------------------------------------------------
    # 6. LayerNorm (用于 Add & Norm)
    # ---------------------------------------------------------
    # 这里的输入通常需要转置以匹配 BN/LN 的维度预期，或者使用 CrypTen 的 LayerNorm 实现
    # 为简单起见，使用 BatchNorm1d 模拟开销（通信量近似）
    ln = crypten.nn.BatchNorm1d(HIDDEN).to(device).encrypt()
    x_ln = x_h.transpose(1, 2) 
    results.append(run_benchmark("LayerNorm", ln, [x_ln], device))

    # ---------------------------------------------------------
    # 汇总输出 (BERT-Base 占比分析)
    # ---------------------------------------------------------
    if rank == 0:
        df = pd.DataFrame(results)
        print("\n" + "="*40)
        print("   Component Micro-Benchmarks (TTP)")
        print("="*40)
        print(df.to_string(index=False))
        
        # 提取单次开销
        def get_stat(name, col):
            return df.loc[df['Component'] == name, col].values[0]

        # BERT-Base 包含 12 个 Encoder Layer
        # 每个 Layer 包含:
        #   Attention: 
        #       - 4个 Linear(768->768) [Q, K, V, Output]
        #       - 1个 Softmax
        #   FFN:
        #       - 1个 Linear(768->3072) [Up]
        #       - 1个 Linear(3072->768) [Down]
        #       - 1个 GELU
        #   Norm:
        #       - 2个 LayerNorm
        
        c_lin_hh = get_stat("Linear (768->768)", "Comm(MB)")
        c_lin_hf = get_stat("Linear (768->3072)", "Comm(MB)")
        c_lin_fh = get_stat("Linear (3072->768)", "Comm(MB)")
        c_softmax = get_stat("Softmax (Attn)", "Comm(MB)")
        c_gelu = get_stat("GELU (3072)", "Comm(MB)")
        c_ln = get_stat("LayerNorm", "Comm(MB)")

        # 单层 Encoder 的通信量
        layer_comm_linear = (4 * c_lin_hh) + c_lin_hf + c_lin_fh
        layer_comm_softmax = c_softmax
        layer_comm_gelu = c_gelu
        layer_comm_ln = 2 * c_ln
        
        layer_total = layer_comm_linear + layer_comm_softmax + layer_comm_gelu + layer_comm_ln
        
        # 12 层总计
        total_bert_comm = layer_total * 12

        print("\n" + "="*40)
        print("   BERT-Base Communication Breakdown")
        print("   (Total 12 Layers Estimate)")
        print("="*40)
        print(f"Total Communication: {total_bert_comm:.2f} MB\n")
        
        if total_bert_comm > 0:
            print(f"Linear (MatMul): {layer_comm_linear*12:.2f} MB ({layer_comm_linear/layer_total:.2%})")
            print(f"Softmax:         {layer_comm_softmax*12:.2f} MB ({layer_comm_softmax/layer_total:.2%})")
            print(f"GELU:            {layer_comm_gelu*12:.2f} MB ({layer_comm_gelu/layer_total:.2%})")
            print(f"LayerNorm:       {layer_comm_ln*12:.2f} MB ({layer_comm_ln/layer_total:.2%})")
        else:
            print("ERROR: Total communication is 0. Check TTP configuration.")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 针对 GPU 环境强制设置 spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. 设置随机种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Launching TTP Benchmark on {DEVICE}...")

    # 3. 手动构建启动器 (避免装饰器 pickle 错误)
    # 必须使用 run_multiprocess(world_size=2) 来模拟 Alice 和 Bob
    launcher = crypten.mpc.context.run_multiprocess(world_size=2)(worker_main)
    
    # 4. 运行
    launcher(DEVICE)