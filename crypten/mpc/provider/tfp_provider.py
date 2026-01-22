#!/usr/bin/env python3

# Modified by Andes Y. L. Kei: Implemented generate_trig_triple, generate_one_hot_pair
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import math
import torch
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element, generate_unsigned_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from .provider import TupleProvider


class TrustedFirstParty(TupleProvider):
    NAME = "TFP"

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        a = generate_random_ring_element(size0, device=device)%100
        b = generate_random_ring_element(size1, device=device)%100

        c = getattr(torch, op)(a, b, *args, **kwargs)

        a = ArithmeticSharedTensor(a, precision=0, src=0)
        b = ArithmeticSharedTensor(b, precision=0, src=0)
        c = ArithmeticSharedTensor(c, precision=0, src=0)

        return a, b, c

    def square(self, size, device=None):
        """Generate square double of given size"""
        r = generate_random_ring_element(size, device=device)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        stacked = torch_stack([r, r2])
        stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    def cube_2(self, size, device=None, mode="cube"):
        r = torch.randn(size, device=device)%10000
        r2 = r.mul(r)
        r3 = r2.mul(r)

        r0 = ArithmeticSharedTensor(r,src=0)
        r = ArithmeticSharedTensor(r, precision=0,src=0)
        r2 = ArithmeticSharedTensor(r2, precision=0, src=0)
        r3 = ArithmeticSharedTensor(r3, precision=0, src=0)
        return r0,r,r2,r3
    
    def cube_3(self, size, device=None, mode="cube"):
        r_plain = torch.randn(size, device=device)
        
        # 提前计算好 r^2, r^3
        r2_plain = r_plain ** 2
        r3_plain = r_plain ** 3
        
        # 2. 加密成 MPC 张量
        # crypten.cryptensor 会自动加上 Scale (默认 2^16)，保证和你的输入 x 同源
        r = crypten.cryptensor(r_plain, src=0)
        r2 = crypten.cryptensor(r2_plain, src=0)
        r3 = crypten.cryptensor(r3_plain, src=0)
        return r, r2, r3
    def cube(self, size, device=None, mode="cube"):
        """
        Generate triples.
        mode="cube": Returns (r, r^2, r^3)
        mode="xy_square": Returns (l1, l2, l1*l2, l2^2, l1*l2^2)
        """
        if mode == "cube":
            r = generate_random_ring_element(size, device=device)
            # r= torch.randint(-10000,10000,size, device=device)
            r2 = r.mul(r)
            r3 = r2.mul(r)
            
            stacked = torch_stack([r, r2, r3])
            stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
            return stacked[0], stacked[1], stacked[2]

        elif mode == "xy_square":
            l1 = generate_random_ring_element(size, device=device)
            l2 = generate_random_ring_element(size, device=device)
            
            l1_l2 = l1.mul(l2)
            l2_sq = l2.mul(l2)
            l1_l2_sq = l1.mul(l2_sq)

            stacked = torch_stack([l1, l2, l1_l2, l2_sq, l1_l2_sq])
            stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
            
            # 按顺序返回: l1, l2, l1_l2, l2_sq, l1_l2_sq
            return stacked[0], stacked[1], stacked[2], stacked[3], stacked[4]
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")
    
    def generate_hybrid_triple(self, size, period, terms, device=None):
        """
        生成混合元组，同时支持傅里叶级数和立方计算。
        返回: 
        - t: 随机掩码 (也是多项式的 r)
        - u, v: sin(kt), cos(kt) (用于 Fourier)
        - t2, t3: t^2, t^3 (用于 x^3 计算)
        """
        if period > 50.0:
            scale = 100.0 
        else:
            scale = period
            
        t_plain = torch.rand(size, device=device) * scale
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t_plain for i in k])
        u_plain, v_plain = torch.sin(tk), torch.cos(tk)
        t2_plain = t_plain.square()
        t3_plain = t2_plain * t_plain

        t = ArithmeticSharedTensor(t_plain, src=0)
        u = ArithmeticSharedTensor(u_plain, src=0)
        v = ArithmeticSharedTensor(v_plain, src=0)
        t2 = ArithmeticSharedTensor(t2_plain, src=0)
        t3 = ArithmeticSharedTensor(t3_plain, src=0)

        return t, u, v, t2, t3

    
    def generate_cmp_aux(self, size, device=None):
            """
            生成 Protocol 4 (CMP+) 所需的辅助随机数 (a, b, r, c)。
            对应论文中 Offline Phase。
            """
            a_abs = torch.rand(size, device=device) * 10 + 1 
            
            b_abs = torch.rand(size, device=device) * (a_abs * 1e-9) 
            
            sign = (torch.rand(size, device=device) > 0.5).float() * 2 - 1
            
            a = a_abs * sign
            b = b_abs * sign

            r = sign 
            c = (1 - sign) / 2
            
            enc_a = ArithmeticSharedTensor(a, src=0)
            enc_b = ArithmeticSharedTensor(b, src=0)
            enc_r = ArithmeticSharedTensor(r, src=0)
            enc_c = ArithmeticSharedTensor(c, src=0)
            
            return enc_a, enc_b, enc_r, enc_c

    def generate_trig_triple(self, size, period, terms, device=None):
        """Generate trigonometric triple of given size"""
        t = torch.rand(size, device=device) * period
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch_stack([i * t for i in k])
        u, v = torch.sin(tk), torch.cos(tk)

        t = ArithmeticSharedTensor(t, src=0)
        u = ArithmeticSharedTensor(u, src=0)
        v = ArithmeticSharedTensor(v, src=0)
        return t, u, v

    def generate_one_hot_pair(self, size, length, device=None):
        """Generate one hot encoding of given size (of output) and length (of one hot vector)"""
        r = generate_unsigned_random_ring_element(size, ring_size=length, device=device)
        v = torch.nn.functional.one_hot(r, num_classes=length)

        r = crypten.cryptensor(r, device=device)
        v = crypten.cryptensor(v, device=device)
        return r, v

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        a = generate_kbit_random_tensor(size0, device=device)
        b = generate_kbit_random_tensor(size1, device=device)
        c = a & b

        a = BinarySharedTensor(a, src=0)
        b = BinarySharedTensor(b, src=0)
        c = BinarySharedTensor(c, src=0)

        return a, b, c

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        num_parties = comm.get().get_world_size()
        r = [
            generate_random_ring_element(size, device=device)
            for _ in range(num_parties)
        ]
        theta_r = count_wraps(r)

        shares = comm.get().scatter(r, 0)
        r = ArithmeticSharedTensor.from_shares(shares, precision=0)
        theta_r = ArithmeticSharedTensor(theta_r, precision=0, src=0)

        return r, theta_r

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        # generate random bit
        r = generate_kbit_random_tensor(size, bitlength=1, device=device)

        rA = ArithmeticSharedTensor(r, precision=0, src=0)
        rB = BinarySharedTensor(r, src=0)

        return rA, rB
    