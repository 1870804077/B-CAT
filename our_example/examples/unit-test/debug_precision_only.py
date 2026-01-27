import torch
import crypten
import logging

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu" # 如果CUDA报错，尝试切回CPU调试

def debug_sigmoid_single_run():
    # 1. 初始化 (单进程模式，方便看Print)
    crypten.init()
    
    # 2. 暴力覆盖所有可能的配置名
    # Crypten 标准配置名是 sigmoid_method
    crypten.cfg.functions.sigmoid_method = "newer_debug"
    # 为了防止你改了源码叫这个名字，也加上
    crypten.cfg.functions.sigmoid_tanh_method = "newer_debug"
    
    print(f"\n{'='*40}")
    print(f"🚀 开始单进程精度调试")
    print(f"当前配置: sigmoid_method = {crypten.cfg.functions.sigmoid_method}")
    print(f"{'='*40}\n")

    # 3. 构造极小数据集 (方便人眼观察)
    # 包含 0, 正数, 负数, 大数
    x_plain = torch.tensor([0.0, 0.5, -0.5, 2.0, -2.0, 6.0], device=DEVICE)
    y_true = torch.sigmoid(x_plain)
    
    print(f"[Input] x: {x_plain.tolist()}")
    print(f"[True ] y: {y_true.tolist()}")

    # 4. 加密并计算
    x_enc = crypten.cryptensor(x_plain)
    
    # 尝试调用，传入 k1, k2
    # 注意：如果你的源码没改好 kwargs 传递，这里可能会报错
    k1, k2 = 1, 8
    print(f"\n[Action] 调用 .sigmoid(k1={k1}, k2={k2})...")
    
    try:
        # 显式传入 method 试图覆盖 (如果你的接口支持)
        # 如果你的接口不支持 method 参数，请删掉 method="newer_debug"
        y_enc = x_enc.sigmoid(k1=k1, k2=k2) 
    except Exception as e:
        print(f"❌ 调用出错: {e}")
        return

    # 5. 解密对比
    y_out = y_enc.get_plain_text()
    print(f"\n[MPC  ] y: {y_out.tolist()}")
    
    diff = (y_true - y_out).abs()
    print(f"\n[Diff ] Max Error: {diff.max().item():.8f}")
    print(f"[Diff ] Avg Error: {diff.mean().item():.8f}")

    if diff.max().item() > 0.1:
        print("\n❌ 误差过大！请检查 newer_debug 中的 linear_term 或 poly_body 系数是否正确。")
    else:
        print("\n✅ 精度正常！")

if __name__ == "__main__":
    debug_sigmoid_single_run()