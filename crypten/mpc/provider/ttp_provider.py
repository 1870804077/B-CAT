#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import crypten
import crypten.communicator as comm
import math
import torch
import torch.distributed as dist
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps
from crypten.encoder import FixedPointEncoder
from crypten.mpc.mpc import MPCTensor
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from .provider import TupleProvider


TTP_FUNCTIONS = ["additive", "square", "binary", "wraps", "B2A"]


class TrustedThirdParty(TupleProvider):
    NAME = "TTP"

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        generator = TTPClient.get().get_generator(device=device)
        a = generate_random_ring_element(size0, generator=generator, device=device)%100
        b = generate_random_ring_element(size1, generator=generator, device=device)%100
        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request(
                "additive", device, size0, size1, op, *args, **kwargs
            )
        else:
            # TODO: Compute size without executing computation
            c_size = getattr(torch, op)(a, b, *args, **kwargs).size()
            c = generate_random_ring_element(c_size, generator=generator, device=device)

        a = ArithmeticSharedTensor.from_shares(a, precision=0)
        b = ArithmeticSharedTensor.from_shares(b, precision=0)
        c = ArithmeticSharedTensor.from_shares(c, precision=0)

        return a, b, c
    def generate_3smp_triple(self, size, device=None):
        """
        [客户端] 生成通用 3SMP 所需的 7 元组:
        Returns: (a, b, c, ab, bc, ca, abc)
        用于计算 x * y * z
        """
        generator = TTPClient.get().get_generator(device=device)
        num_outputs = 7  # a, b, c, ab, bc, ca, abc

        if comm.get().get_rank() == 0:
            # Rank 0: 向 Server 请求
            stacked = TTPClient.get().ttp_request("generate_3smp_triple", device, size)
            
            # 如果 Server 返回的是 CPU tensor，需要转回 device
            if device is not None:
                stacked = stacked.to(device)

            shares_list = list(stacked.split(1, dim=0))
            shares_list = [s.squeeze(0) for s in shares_list]
        else:
            # Rank > 0: 本地生成 7 个随机 shares
            shares_list = [
                generate_random_ring_element(size, generator=generator, device=device)
                for _ in range(num_outputs)
            ]

        # 转换为 ArithmeticSharedTensor
        results = [
            ArithmeticSharedTensor.from_shares(s, precision=0) 
            for s in shares_list
        ]
        
        # 返回 tuple: (a, b, c, ab, bc, ca, abc)
        return tuple(results)
    def square(self, size, device=None):
        """Generate square double of given size"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request r2 from TTP
            r2 = TTPClient.get().ttp_request("square", device, size)
        else:
            r2 = generate_random_ring_element(size, generator=generator, device=device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        r2 = ArithmeticSharedTensor.from_shares(r2, precision=0)
        return r, r2
    def cube(self, size, device=None, mode="cube"):
        """
        Generate triples for x^3 or xy^2.
        mode="cube": Returns (r, r^2, r^3)
        mode="xy_square": Returns (l1, l2, l1*l2, l2^2, l1*l2^2)
        """
        generator = TTPClient.get().get_generator(device=device)
        
        # 定义输出张量的数量
        if mode == "cube":
            num_outputs = 3
        elif mode == "xy_square":
            num_outputs = 5
        else:
            raise ValueError(f"Unknown cube mode: {mode}")

        if comm.get().get_rank() == 0:
            stacked = TTPClient.get().ttp_request("cube", device, size, mode)
            shares_list = list(stacked.split(1, dim=0))
            shares_list = [s.squeeze(0) for s in shares_list]
        else:
            # 其他参与方本地生成随机 share
            shares_list = [
                generate_random_ring_element(size, generator=generator, device=device)
                for _ in range(num_outputs)
            ]

        # 转换为 ArithmeticSharedTensor
        results = [
            ArithmeticSharedTensor.from_shares(s, precision=0) 
            for s in shares_list
        ]
        
        # 解包返回
        return tuple(results)
    def cube_1(self, size, device=None, mode="cube"):
        """
        Generate triples for x^3 or xy^2.
        """
        generator = TTPClient.get().get_generator(device=device)
        
        if mode == "cube":
            num_outputs = 3
        elif mode == "xy_square":
            num_outputs = 5
        else:
            raise ValueError(f"Unknown cube mode: {mode}")

        if comm.get().get_rank() == 0:
            # Rank 0: 请求 TTP
            stacked = TTPClient.get().ttp_request("cube_1", device, size, mode)
            
            if device is not None:
                stacked = stacked.to(device) # 转回 GPU

            shares_list = list(stacked.split(1, dim=0))
            shares_list = [s.squeeze(0) for s in shares_list]
        else:
            # Rank > 0: 本地生成
            shares_list = [
                generate_random_ring_element(size, generator=generator, device=device)
                for _ in range(num_outputs) # 现在 num_outputs 是可见的
            ]

        # 封装返回
        results = [
            ArithmeticSharedTensor.from_shares(s, precision=0) 
            for s in shares_list
        ]
        
        return tuple(results)
    def generate_gelu_offline_batch(self, input_size, double_size, period, terms, device=None):
        """
        [客户端] 离线数据批发接口
        发送一次请求，获取所有离线数据
        """
        generator = TTPClient.get().get_generator(device=device)
        
        # 预先计算各部分的元素数量 (numel)
        num_double = double_size.numel() if isinstance(double_size, torch.Size) else torch.tensor(double_size).prod().item()
        num_input = input_size.numel() if isinstance(input_size, torch.Size) else torch.tensor(input_size).prod().item()
        
        # Hybrid Triple 的大小计算
        # t(1) + u(terms) + v(terms) + t2(1) + t3(1) = 3 + 2*terms
        num_hybrid_total = (3 + 2 * terms) * num_input

        # === 1. 本地生成 Triples 的 a, b 分量 (PRSS) ===
        # TTP 模式下，Triple 的 a, b 是由客户端本地生成的，TTP Server 只给 c
        # 必须按顺序生成，以保持 RNG 同步
        
        # Triple 1 (CMP)
        u_cmp_a = generate_random_ring_element(double_size, generator=generator, device=device)
        u_cmp_b = generate_random_ring_element(double_size, generator=generator, device=device)
        
        # Triple 2 (GeLU)
        u_gelu_a = generate_random_ring_element(input_size, generator=generator, device=device)
        u_gelu_b = generate_random_ring_element(input_size, generator=generator, device=device)
        
        # Triple 3 (Final)
        u_final_a = generate_random_ring_element(double_size, generator=generator, device=device)
        u_final_b = generate_random_ring_element(double_size, generator=generator, device=device)
        
        # === 2. 发送/模拟 TTP 请求 ===
        if comm.get().get_rank() == 0:
            # Rank 0 向服务器请求“大礼包”
            packed_res = TTPClient.get().ttp_request(
                "generate_gelu_offline_batch", device, input_size, double_size, period, terms
            )
            
            # === 3. 拆包 (Slicing) ===
            cursor = 0
            
            # Part 1: CMP Aux (4 * double_size)
            end = cursor + 4 * num_double
            res_cmp_aux = packed_res[cursor:end].view(4, *double_size)
            cursor = end
            
            # Part 2: Triple 1 c (double_size)
            end = cursor + num_double
            res_trip1_c = packed_res[cursor:end].view(double_size)
            cursor = end
            
            # Part 3: Hybrid (total hybrid size)
            end = cursor + num_hybrid_total
            # Hybrid 需要进一步细拆，先保持 flat 或者按照 hybrid 的逻辑 view
            # Server 端 hybrid 是 cat 起来的: t(1), u(terms), v(terms), t2(1), t3(1)
            # 为了方便，我们这里先拿到整块，后面再细分
            # 注意：TTP Server 返回的 hybrid 是 shape (3+2terms, *input_size)
            res_hybrid = packed_res[cursor:end].view(3 + 2 * terms, *input_size)
            cursor = end
            
            # Part 4: Triple 2 c (input_size)
            end = cursor + num_input
            res_trip2_c = packed_res[cursor:end].view(input_size)
            cursor = end
            
            # Part 5: Triple 3 c (double_size)
            end = cursor + num_double
            res_trip3_c = packed_res[cursor:end].view(double_size)
            
            # === 4. 组装 AST 对象 ===
            # CMP Aux
            shares = [s for s in res_cmp_aux]
            a_all, b_all, r_all, c_all = shares
            
            # Triple 1 (CMP)
            u_cmp_c = res_trip1_c
            
            # Hybrid
            # 解构 res_hybrid
            t_erf = res_hybrid[0]
            u_erf = res_hybrid[1 : 1 + terms]
            v_erf = res_hybrid[1 + terms : 1 + 2 * terms]
            t2_erf = res_hybrid[1 + 2 * terms]
            t3_erf = res_hybrid[1 + 2 * terms + 1]
            
            # Triple 2 (GeLU)
            u_gelu_c = res_trip2_c
            
            # Triple 3 (Final)
            u_final_c = res_trip3_c
            
        else:
            # Rank > 0: 生成本地随机数占位 (PRSS)
            # TTPClient 的逻辑是：Client 本地生成 share，Server 生成 share，两者相加 = 0 或 真值
            # 对于 TTP Request 的部分，Rank > 0 需要生成对应的随机 mask
            
            # CMP Aux
            a_all = generate_random_ring_element(double_size, generator=generator, device=device)
            b_all = generate_random_ring_element(double_size, generator=generator, device=device)
            r_all = generate_random_ring_element(double_size, generator=generator, device=device)
            c_all = generate_random_ring_element(double_size, generator=generator, device=device)
            
            # Triples c 分量
            u_cmp_c = generate_random_ring_element(double_size, generator=generator, device=device)
            
            # Hybrid
            hybrid_shape = (3 + 2 * terms, *input_size)
            res_hybrid = generate_random_ring_element(hybrid_shape, generator=generator, device=device)
            t_erf = res_hybrid[0]
            u_erf = res_hybrid[1 : 1 + terms]
            v_erf = res_hybrid[1 + terms : 1 + 2 * terms]
            t2_erf = res_hybrid[1 + 2 * terms]
            t3_erf = res_hybrid[1 + 2 * terms + 1]
            
            u_gelu_c = generate_random_ring_element(input_size, generator=generator, device=device)
            u_final_c = generate_random_ring_element(double_size, generator=generator, device=device)

        # === 5. 封装返回 ===
        # Helper to wrap AST
        def to_ast(x): return ArithmeticSharedTensor.from_shares(x, precision=0)
        
        # CMP Aux
        aux_ret = (to_ast(a_all), to_ast(b_all), to_ast(r_all), to_ast(c_all))
        
        # Triples (a, b from local, c from TTP)
        trip1_ret = (to_ast(u_cmp_a), to_ast(u_cmp_b), to_ast(u_cmp_c))
        trip2_ret = (to_ast(u_gelu_a), to_ast(u_gelu_b), to_ast(u_gelu_c))
        trip3_ret = (to_ast(u_final_a), to_ast(u_final_b), to_ast(u_final_c))
        
        # Hybrid
        hybrid_ret = (to_ast(t_erf), to_ast(u_erf), to_ast(v_erf), to_ast(t2_erf), to_ast(t3_erf))
        
        return aux_ret, trip1_ret, hybrid_ret, trip2_ret, trip3_ret
    
    # def generate_hybrid_triple(self, size, period, terms, device=None):
    #     """
    #     Generate hybrid triples (Fourier + Cube).
    #     Returns: t, u, v, t2, t3
    #     """
    #     generator = TTPClient.get().get_generator(device=device)
        
    #     uv_shape = (terms,) + size
        
    #     if comm.get().get_rank() == 0:
    #         cat_res = TTPClient.get().ttp_request("generate_hybrid_triple", device, size, period, terms)
            
    #         idx_u_start = 1
    #         idx_v_start = 1 + terms
    #         idx_t2 = 1 + 2 * terms
    #         idx_t3 = idx_t2 + 1
            
    #         t = cat_res[0]
    #         u = cat_res[idx_u_start:idx_v_start]
    #         v = cat_res[idx_v_start:idx_t2]
    #         t2 = cat_res[idx_t2]
    #         t3 = cat_res[idx_t3]
    #     else:
    #         t = generate_random_ring_element(size, generator=generator, device=device)
    #         u = generate_random_ring_element(uv_shape, generator=generator, device=device)
    #         v = generate_random_ring_element(uv_shape, generator=generator, device=device)
    #         t2 = generate_random_ring_element(size, generator=generator, device=device)
    #         t3 = generate_random_ring_element(size, generator=generator, device=device)

    #     return (
    #         ArithmeticSharedTensor.from_shares(t),
    #         ArithmeticSharedTensor.from_shares(u),
    #         ArithmeticSharedTensor.from_shares(v),
    #         ArithmeticSharedTensor.from_shares(t2),
    #         ArithmeticSharedTensor.from_shares(t3)
    #     )
    def generate_hybrid_triple(self, size, period, terms, device=None):

        generator = TTPClient.get().get_generator(device=device)
        
        total_channels = 1 + terms + terms + 1 + 1
        
        concat_shape = (total_channels, *size)
        
        if comm.get().get_rank() == 0:
            cat_res = TTPClient.get().ttp_request("generate_hybrid_triple", device, size, period, terms)
            
            # [关键] 搬回 GPU
            if device is not None:
                cat_res = cat_res.to(device)
            
        else:
            cat_res = generate_random_ring_element(concat_shape, generator=generator, device=device)
            
        idx_u_start = 1
        idx_v_start = 1 + terms
        idx_t2 = 1 + 2 * terms
        idx_t3 = idx_t2 + 1
        
        t = cat_res[0]
        u = cat_res[idx_u_start:idx_v_start]
        v = cat_res[idx_v_start:idx_t2]
        t2 = cat_res[idx_t2]
        t3 = cat_res[idx_t3]

        # 修正维度 (t, t2, t3 在 cat 时多了个维度，这里还原回 size)
        if t.shape != size: t = t.reshape(size)
        if t2.shape != size: t2 = t2.reshape(size)
        if t3.shape != size: t3 = t3.reshape(size)

        return (
            ArithmeticSharedTensor.from_shares(t),
            ArithmeticSharedTensor.from_shares(u),
            ArithmeticSharedTensor.from_shares(v),
            ArithmeticSharedTensor.from_shares(t2),
            ArithmeticSharedTensor.from_shares(t3)
        )
    
    def generate_hybrid_triple_1(self, size, period, terms, device=None):

        generator = TTPClient.get().get_generator(device=device)
        
        total_channels = 1 + terms + terms
        
        concat_shape = (total_channels, *size)
        
        if comm.get().get_rank() == 0:
            cat_res = TTPClient.get().ttp_request("generate_hybrid_triple_1", device, size, period, terms)
            
            # [关键] 搬回 GPU
            if device is not None:
                cat_res = cat_res.to(device)
            
        else:
            cat_res = generate_random_ring_element(concat_shape, generator=generator, device=device)
            
        idx_u_start = 1
        idx_v_start = 1 + terms
        idx_t2 = 1 + 2 * terms
        
        t = cat_res[0]
        u = cat_res[idx_u_start:idx_v_start]
        v = cat_res[idx_v_start:idx_t2]

        if t.shape != size: t = t.reshape(size)

        return (
            ArithmeticSharedTensor.from_shares(t),
            ArithmeticSharedTensor.from_shares(u),
            ArithmeticSharedTensor.from_shares(v),
        )
    
    def generate_reciprocal_offline(self, size, period_s, period_l, beta_s_len, beta_l_len, device=None):
        """
        [Client 端] 一站式请求 Reciprocal 所需的所有离线数据
        """
        generator = TTPClient.get().get_generator(device=device)
        
        # 计算所需的 tensor 总数，用于本地生成占位符
        # Hybrid (5) * 2 + CMP (4) + Newton (5) * 3 = 10 + 4 + 15 = 29
        total_tensors = 29
        
        if comm.get().get_rank() == 0:
            # Rank 0: 向 Server 请求大礼包
            # 注意: 这里调用的是 Server 端的 "reciprocal_offline"
            packed_res = TTPClient.get().ttp_request(
                "reciprocal_offline", device, size, period_s, period_l, beta_s_len, beta_l_len
            )
            
            # 拿到的是一个巨大的 Stacked Tensor，在第一维切分
            shares_list = [s.squeeze(0) for s in packed_res.split(1, dim=0)]
        else:
            # Rank > 0: 本地生成随机数
            shares_list = [
                generate_random_ring_element(size, generator=generator, device=device)
                for _ in range(total_tensors)
            ]

        # 统一转为 AST
        results = [
            ArithmeticSharedTensor.from_shares(s, precision=0) 
            for s in shares_list
        ]
        
        # 按顺序解包返回
        # 1. Hybrid Small (5个)
        res_s = results[0:5]
        # 2. Hybrid Large (5个)
        res_l = results[5:10]
        # 3. CMP Aux (4个)
        res_cmp = results[10:14]
        # 4. Newton Triples (3组，每组5个)
        res_newton = []
        base = 14
        for i in range(3):
            res_newton.append(results[base + i*5 : base + (i+1)*5])
            
        return res_s, res_l, res_cmp, res_newton
    
    def generate_cmp_aux(self, size, device=None):
        """
        Generate auxiliary parameters for CMP protocol: a, b, r, c
        """
        generator = TTPClient.get().get_generator(device=device)
        
        if comm.get().get_rank() == 0:
            # TTP Server 返回 stack([a, b, r, c])
            stacked = TTPClient.get().ttp_request("generate_cmp_aux", device, size)
            shares = [s.squeeze(0) for s in stacked.split(1, dim=0)]
            a, b, r, c = shares[0], shares[1], shares[2], shares[3]
        else:
            a = generate_random_ring_element(size, generator=generator, device=device)
            b = generate_random_ring_element(size, generator=generator, device=device)
            r = generate_random_ring_element(size, generator=generator, device=device)
            c = generate_random_ring_element(size, generator=generator, device=device)
            
        return (
            ArithmeticSharedTensor.from_shares(a, precision=0), # precision=0 因为 TTP 返回的已经是 encoded 整数
            ArithmeticSharedTensor.from_shares(b, precision=0),
            ArithmeticSharedTensor.from_shares(r, precision=0),
            ArithmeticSharedTensor.from_shares(c, precision=0)
        )
    
    def generate_trig_triple(self, size, period, terms, device=None):
        """Generate trigonometric triple of given size"""
        generator = TTPClient.get().get_generator(device=device)
        uv_shape = (terms,) + size
        if comm.get().get_rank() == 0:
            tuv = TTPClient.get().ttp_request("generate_trig_triple", device, size, period, terms)
            t, u, v = tuv[0], tuv[1:1 + terms], tuv[1 + terms:]
        else:
            t = generate_random_ring_element(size, generator=generator, device=device)
            u = generate_random_ring_element(uv_shape, generator=generator, device=device)
            v = generate_random_ring_element(uv_shape, generator=generator, device=device)
        t = ArithmeticSharedTensor.from_shares(t)
        u = ArithmeticSharedTensor.from_shares(u)
        v = ArithmeticSharedTensor.from_shares(v)
        return t, u, v
    
    def generate_one_hot_pair(self, size, length, device=None):
        generator = TTPClient.get().get_generator(device=device)
        v_shape = (*size,) + (length,)
        r = generate_random_ring_element(size, generator=generator, device=device) % length
        if comm.get().get_rank() == 0:
            v = TTPClient.get().ttp_request("generate_one_hot_pair", device, size, length)
        else:
            v = generate_random_ring_element(v_shape, generator=generator, device=device)
        r = MPCTensor.from_shares(r, precision=0)
        v = MPCTensor.from_shares(v, precision=0)
        return r, v

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate binary triples of given size"""
        generator = TTPClient.get().get_generator(device=device)

        a = generate_kbit_random_tensor(size0, generator=generator, device=device)
        b = generate_kbit_random_tensor(size1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request("binary", device, size0, size1)
        else:
            size2 = torch.broadcast_tensors(a, b)[0].size()
            c = generate_kbit_random_tensor(size2, generator=generator, device=device)

        # Stack to vectorize scatter function
        a = BinarySharedTensor.from_shares(a)
        b = BinarySharedTensor.from_shares(b)
        c = BinarySharedTensor.from_shares(c)
        return a, b, c

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        generator = TTPClient.get().get_generator(device=device)

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request theta_r from TTP
            theta_r = TTPClient.get().ttp_request("wraps", device, size)
        else:
            theta_r = generate_random_ring_element(
                size, generator=generator, device=device
            )

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        theta_r = ArithmeticSharedTensor.from_shares(theta_r, precision=0)
        return r, theta_r

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        generator = TTPClient.get().get_generator(device=device)

        # generate random bit
        rB = generate_kbit_random_tensor(
            size, bitlength=1, generator=generator, device=device
        )

        if comm.get().get_rank() == 0:
            # Request rA from TTP
            rA = TTPClient.get().ttp_request("B2A", device, size)
        else:
            rA = generate_random_ring_element(size, generator=generator, device=device)

        rA = ArithmeticSharedTensor.from_shares(rA, precision=0)
        rB = BinarySharedTensor.from_shares(rB)
        return rA, rB

    @staticmethod
    def _init():
        TTPClient._init()

    @staticmethod
    def uninit():
        TTPClient.uninit()


class TTPClient:
    __instance = None

    class __TTPClient:
        """Singleton class"""

        def __init__(self):
            # Initialize connection
            self.ttp_group = comm.get().ttp_group
            self.comm_group = comm.get().ttp_comm_group
            self._setup_generators()
            logging.info(f"TTPClient {comm.get().get_rank()} initialized")

        def _setup_generators(self):
            """Setup RNG generator shared between each party (client) and the TTPServer"""
            seed = torch.empty(size=(), dtype=torch.long)
            dist.irecv(
                tensor=seed, src=comm.get().get_ttp_rank(), group=self.ttp_group
            ).wait()
            dist.barrier(group=self.ttp_group)

            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(seed.item())

            if torch.cuda.is_available():
                self.generator_cuda = torch.Generator(device="cuda")
                self.generator_cuda.manual_seed(seed.item())
            else:
                self.generator_cuda = None

        def get_generator(self, device=None):
            if device is None:
                device = "cpu"
            device = torch.device(device)
            if device.type == "cuda":
                return self.generator_cuda
            else:
                return self.generator

        def ttp_request(self, func_name, device, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            if device is not None:
                device = str(device)
            message = {
                "function": func_name,
                "device": device,
                "args": args,
                "kwargs": kwargs,
            }
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            result = torch.empty(size, dtype=torch.long, device=device)
            comm.get().broadcast(result, ttp_rank, self.comm_group)

            return result

    @staticmethod
    def _init():
        """Initializes a Trusted Third Party client that sends requests"""
        if TTPClient.__instance is None:
            TTPClient.__instance = TTPClient.__TTPClient()

    @staticmethod
    def uninit():
        """Uninitializes a Trusted Third Party client"""
        del TTPClient.__instance
        TTPClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the TTPClient"""
        if TTPClient.__instance is None:
            raise RuntimeError("TTPClient is not initialized")

        return TTPClient.__instance


class TTPServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes a Trusted Third Party server that receives requests"""
        # Initialize connection
        crypten.init()
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cpu"
        self._setup_generators()
        ttp_rank = comm.get().get_ttp_rank()

        logging.info("TTPServer Initialized")
        try:
            while True:
                # Wait for next request from client
                message = comm.get().recv_obj(0, self.ttp_group)
                logging.info("Message received: %s" % message)

                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return

                function = message["function"]
                device = message["device"]
                args = message["args"]
                kwargs = message["kwargs"]

                self.device = device

                result = getattr(self, function)(*args, **kwargs)

                comm.get().send_obj(result.size(), 0, self.ttp_group)
                comm.get().broadcast(result, ttp_rank, self.comm_group)
        except RuntimeError as err:
            logging.info("Encountered Runtime error. TTPServer shutting down:")
            logging.info(f"{err}")
    def generate_3smp_triple(self, size):
        """
        [服务端] 生成通用 3SMP 三元组 (7-tuple)
        Returns: Stacked tensor [a, b, c, ab, bc, ca, abc]
        """
        # 1. 在 GPU 上生成基础随机数 (必须在 self.device 上以同步随机种子)
        a_gpu = self._get_additive_PRSS(size)
        b_gpu = self._get_additive_PRSS(size)
        c_gpu = self._get_additive_PRSS(size)

        # 2. 在 GPU 上计算所有乘积组合 (利用 CUDA 加速)
        ab_gpu = a_gpu.mul(b_gpu)
        bc_gpu = b_gpu.mul(c_gpu)
        ca_gpu = c_gpu.mul(a_gpu)
        abc_gpu = ab_gpu.mul(c_gpu)

        # 3. [关键步骤] 立即转移到 CPU
        # 避免在 GPU 上进行复杂的 stack 操作，防止分布式环境下的非法内存访问
        parts_cpu = [
            a_gpu.cpu(), b_gpu.cpu(), c_gpu.cpu(),
            ab_gpu.cpu(), bc_gpu.cpu(), ca_gpu.cpu(), abc_gpu.cpu()
        ]

        # 4. 在 CPU 上堆叠
        stacked_cpu = torch.stack(parts_cpu)

        # 5. 生成 Mask (GPU -> CPU) 并进行 PRSS 减法
        # mask 必须在 GPU 上生成以匹配 shape 和 seed，然后转 CPU 运算
        mask_gpu = self._get_additive_PRSS(stacked_cpu.shape, remove_rank=True)
        stacked_cpu -= mask_gpu.cpu()

        # 6. 返回结果 (TTPClient 会处理 device 搬运)
        return stacked_cpu        
    def reciprocal_offline(self, size, period_s, period_l, beta_s_len, beta_l_len):
        """
        [Server 端 - 修正版]
        生成 CPU 数据以防崩溃，但最后必须转回 self.device 以匹配客户端的协议。
        """
        from crypten.encoder import FixedPointEncoder
        encoder = FixedPointEncoder()
        
        # 1. 生成 Hybrid Triples (GPU)
        hybrid_s_gpu = self.generate_hybrid_triple(size, period_s, beta_s_len)
        hybrid_l_gpu = self.generate_hybrid_triple(size, period_l, beta_l_len)
        
        # 2. 生成 CMP Aux (GPU -> CPU)
        # 显式生成
        a = torch.rand(size, device=self.device) * 10 + 100 
        b = torch.rand(size, device=self.device) * (a * 1e-9)
        r = torch.ones(size, device=self.device)
        c = torch.zeros(size, device=self.device)
        
        cmp_aux_gpu = torch.stack([
            encoder.encode(a), encoder.encode(b),
            encoder.encode(r), encoder.encode(c)
        ])
        cmp_aux_gpu -= self._get_additive_PRSS(cmp_aux_gpu.shape, remove_rank=True)
        
        # 3. 生成 Newton Triples (GPU 生成 -> CPU 堆叠)
        newton_parts_cpu = []
        for _ in range(3):
            l1 = self._get_additive_PRSS(size)
            l2 = self._get_additive_PRSS(size)
            l1_l2 = l1.mul(l2)
            l2_sq = l2.mul(l2)
            l1_l2_sq = l1.mul(l2_sq)
            
            # 立即转 CPU
            newton_parts_cpu.extend([
                l1.cpu(), l2.cpu(), l1_l2.cpu(), l2_sq.cpu(), l1_l2_sq.cpu()
            ])
            
        newton_stack_cpu = torch.stack(newton_parts_cpu)
        # 在 CPU 上做 Mask，避免 GPU stack 崩溃
        # 注意：这里的 PRSS 需要临时用 GPU 生成然后转 CPU 减
        # 为了方便，我们这里简化：直接在 CPU 上减（假设 Server CPU 也有随机源，或者容忍这一步的性能损耗）
        # 正确做法：GPU 生成 Mask -> 转 CPU -> 减
        mask_gpu = self._get_additive_PRSS(newton_stack_cpu.shape, remove_rank=True)
        newton_stack_cpu -= mask_gpu.cpu()
        
        # 4. [终极打包] 拼接 (CPU) -> 转回 (self.device)
        flat_res_cpu = torch.cat([
            hybrid_s_gpu.reshape(-1).cpu(),
            hybrid_l_gpu.reshape(-1).cpu(),
            cmp_aux_gpu.reshape(-1).cpu(),
            newton_stack_cpu.reshape(-1).cpu()
        ])
        
        # [关键修复] 如果客户端请求的是 GPU，我们必须发回 GPU 数据！
        # 此时是一个单一的 Flat Tensor，Broadcast 是安全的。
        return flat_res_cpu.to(self.device)
    
    def generate_gelu_offline_batch(self, input_size, double_size, period, terms):
        """
        [服务端] 离线数据批发接口
        """
        from crypten.encoder import FixedPointEncoder
        encoder = FixedPointEncoder()
        
        # --- 1. 生成 CMP Aux ---
        # 保持在 self.device (CUDA) 上计算以利用并行优势
        a = torch.rand(double_size, device=self.device) * 10 + 100 
        b = torch.rand(double_size, device=self.device) * (a * 1e-9)
        r = torch.ones(double_size, device=self.device)
        c = torch.zeros(double_size, device=self.device)
        
        cmp_aux = torch.stack([
            encoder.encode(a), encoder.encode(b),
            encoder.encode(r), encoder.encode(c)
        ])
        cmp_aux -= self._get_additive_PRSS(cmp_aux.shape, remove_rank=True)
        
        # --- 2. 生成 Triples ---
        trip1_c = self.additive(double_size, double_size, "mul")
        trip2_c = self.additive(input_size, input_size, "mul")
        trip3_c = self.additive(double_size, double_size, "mul")

        # --- 3. 生成 Hybrid Triple ---
        # 这会返回 CUDA Tensor (因为 generate_hybrid_triple 也是在 self.device 上)
        hybrid_res = self.generate_hybrid_triple(input_size, period, terms)
        
        # --- 4. [FIX] 强制转 CPU 再拼接 ---
        # 这一步至关重要！它解决了两个问题：
        # 1. 避免 "Expected all tensors to be on same device" 错误
        # 2. 避免 GPU 上的 OOM 或碎片问题
        packed_result = torch.cat([
            cmp_aux.reshape(-1).cpu(),    # <--- 加上 .cpu()
            trip1_c.reshape(-1).cpu(),    # <--- 加上 .cpu()
            hybrid_res.reshape(-1).cpu(), # <--- 加上 .cpu()
            trip2_c.reshape(-1).cpu(),    # <--- 加上 .cpu()
            trip3_c.reshape(-1).cpu()     # <--- 加上 .cpu()
        ])
        
        return packed_result
    def _setup_generators(self):
        """Create random generator to send to a party"""
        ws = comm.get().get_world_size()

        seeds = [torch.randint(-(2**63), 2**63 - 1, size=()) for _ in range(ws)]
        reqs = [
            dist.isend(tensor=seeds[i], dst=i, group=self.ttp_group) for i in range(ws)
        ]
        self.generators = [torch.Generator(device="cpu") for _ in range(ws)]
        self.generators_cuda = [
            (torch.Generator(device="cuda") if torch.cuda.is_available() else None)
            for _ in range(ws)
        ]

        for i in range(ws):
            self.generators[i].manual_seed(seeds[i].item())
            if torch.cuda.is_available():
                self.generators_cuda[i].manual_seed(seeds[i].item())
            reqs[i].wait()

        dist.barrier(group=self.ttp_group)

    def _get_generators(self, device=None):
        if device is None:
            device = "cpu"
        device = torch.device(device)
        if device.type == "cuda":
            return self.generators_cuda
        else:
            return self.generators

    def _get_additive_PRSS(self, size, remove_rank=False):
        """
        Generates a plaintext value from a set of random additive secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_random_ring_element(size, generator=g, device=g.device)
            result = elem if idx == 0 else result + elem
        return result

    def _get_binary_PRSS(self, size, bitlength=None, remove_rank=None):
        """
        Generates a plaintext value from a set of random binary secret shares
        generated by each party
        """
        gens = self._get_generators(device=self.device)
        if remove_rank:
            gens = gens[1:]
        result = None
        for idx, g in enumerate(gens):
            elem = generate_kbit_random_tensor(
                size, bitlength=bitlength, generator=g, device=g.device
            )
            result = elem if idx == 0 else result ^ elem
        return result

    def additive(self, size0, size1, op, *args, **kwargs):

        # Add all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_additive_PRSS(size0)
        b = self._get_additive_PRSS(size1)

        c = getattr(torch, op)(a, b, *args, **kwargs)

        # Subtract all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c - self._get_additive_PRSS(c.size(), remove_rank=True)
        return c0

    def square(self, size):
        # Add all shares of `r` to get plaintext `r`
        r = self._get_additive_PRSS(size)
        r2 = r.mul(r)
        return r2 - self._get_additive_PRSS(size, remove_rank=True)

    def binary(self, size0, size1):
        # xor all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_binary_PRSS(size0)
        b = self._get_binary_PRSS(size1)

        c = a & b

        # xor all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c ^ self._get_binary_PRSS(c.size(), remove_rank=True)
        return c0

    def cube(self, size, mode="cube"):
        """
        Generate triples for x^3 or xy^2 on TTP Server.
        """
        if mode == "cube":
            # 1. 恢复出 r 的明文 (Sum of all parties' shares)
            r = self._get_additive_PRSS(size)
            
            # 2. 计算 r^2, r^3
            r2 = r.mul(r)
            r3 = r2.mul(r)
            
            # 3. 堆叠
            stacked = torch.stack([r, r2, r3])
            
            # 4. 减去其他方的 shares (PRSS)
            # 注意: _get_additive_PRSS 需要对应 shape
            stacked -= self._get_additive_PRSS(stacked.size(), remove_rank=True)
            
            return stacked

        elif mode == "xy_square":
            l1 = self._get_additive_PRSS(size)
            l2 = self._get_additive_PRSS(size)
            
            l1_l2 = l1.mul(l2)
            l2_sq = l2.mul(l2)
            l1_l2_sq = l1.mul(l2_sq)

            stacked = torch.stack([l1, l2, l1_l2, l2_sq, l1_l2_sq])
            stacked -= self._get_additive_PRSS(stacked.size(), remove_rank=True)
            return stacked
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")
        
    def cube_1(self, size, mode="cube"):
        """
        [ULTIMATE FIX] TTP Server 端生成 Cube 三元组
        策略：GPU 生成 (同步随机数) -> CPU 组装 (防止崩溃)
        """
        if mode == "cube":
            # 1. 在 GPU 上生成基础随机数 r (必须在 GPU 以匹配 Client 的 Seed)
            r_gpu = self._get_additive_PRSS(size)
            
            # 2. 在 GPU 上计算乘方 (利用 GPU 加速)
            r2_gpu = r_gpu.mul(r_gpu)
            r3_gpu = r2_gpu.mul(r_gpu)
            
            # 3. [关键步骤] 立即转移到 CPU
            # 避免在 GPU 上做 stack，这在某些分布式环境下不稳定
            r_cpu = r_gpu.cpu()
            r2_cpu = r2_gpu.cpu()
            r3_cpu = r3_gpu.cpu()
            
            # 4. 在 CPU 上堆叠
            stacked_cpu = torch.stack([r_cpu, r2_cpu, r3_cpu])
            
            # 5. 生成 Mask (GPU) -> 转 CPU -> 减法
            # 我们需要生成一个形状匹配 stacked 的 mask
            # 先计算形状
            stack_shape = stacked_cpu.shape
            
            # 调用 PRSS 生成 mask (这一步会在 GPU 上跑，因为 self.device 是 cuda)
            mask_gpu = self._get_additive_PRSS(stack_shape, remove_rank=True)
            
            # 转 CPU 进行减法
            stacked_cpu -= mask_gpu.cpu()
            
            return stacked_cpu

        elif mode == "xy_square":
            # 1. GPU 生成
            l1_gpu = self._get_additive_PRSS(size)
            l2_gpu = self._get_additive_PRSS(size)
            
            # 2. GPU 计算
            l1_l2_gpu = l1_gpu.mul(l2_gpu)
            l2_sq_gpu = l2_gpu.mul(l2_gpu)
            l1_l2_sq_gpu = l1_gpu.mul(l2_sq_gpu)

            # 3. 转 CPU
            l1_c = l1_gpu.cpu()
            l2_c = l2_gpu.cpu()
            l1_l2_c = l1_l2_gpu.cpu()
            l2_sq_c = l2_sq_gpu.cpu()
            l1_l2_sq_c = l1_l2_sq_gpu.cpu()

            # 4. CPU 堆叠
            stacked_cpu = torch.stack([l1_c, l2_c, l1_l2_c, l2_sq_c, l1_l2_sq_c])
            
            # 5. Mask (GPU -> CPU)
            mask_gpu = self._get_additive_PRSS(stacked_cpu.shape, remove_rank=True)
            stacked_cpu -= mask_gpu.cpu()
            
            return stacked_cpu
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")

    # def generate_hybrid_triple(self, size, period, terms):
    #     """
    #     [服务端] Generate hybrid triples (Fourier + Cube).
    #     Returns Tensor on self.device (CUDA if available)
    #     """
    #     encoder = FixedPointEncoder()
        
    #     # 1. 确定 Scaling
    #     if period > 50.0:
    #         scale = 100.0 
    #     else:
    #         scale = period
            
    #     # 2. 生成明文数据 (Floats)
    #     # 都在 self.device 上生成
    #     t_plain = torch.rand(size, device=self.device) * scale
        
    #     k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
    #     tk = torch.stack([i * t_plain for i in k])
        
    #     u_plain = torch.sin(tk)
    #     v_plain = torch.cos(tk)
    #     t2_plain = t_plain.square()
    #     t3_plain = t2_plain * t_plain

    #     # 3. 编码 (Float -> FixedPoint Integer)
    #     # t, t2, t3 需要 unsqueeze 变成 (1, *size)
    #     t_enc = encoder.encode(t_plain, device=self.device).reshape(1, *size)
    #     u_enc = encoder.encode(u_plain, device=self.device) # (terms, *size)
    #     v_enc = encoder.encode(v_plain, device=self.device) # (terms, *size)
    #     t2_enc = encoder.encode(t2_plain, device=self.device).reshape(1, *size)
    #     t3_enc = encoder.encode(t3_plain, device=self.device).reshape(1, *size)
        
    #     # 4. 拼接 (在 GPU 上进行)
    #     results = torch.cat([t_enc, u_enc, v_enc, t2_enc, t3_enc], dim=0)
        
    #     # 5. 减去其他方的 shares
    #     results -= self._get_additive_PRSS(results.shape, remove_rank=True)

    #     return results

    def generate_hybrid_triple(self, size, period, terms):
        """
        [服务端 - Debugged] 
        输出源头数据，确认 t 是浮点数且 sin/cos 计算正确。
        """
        encoder = FixedPointEncoder()
        
        # 1. 确定 Scaling
        if period > 50.0:
            scale = 100.0 
        else:
            scale = period
            
        # 2. 生成 t (明文浮点数)
        t_plain = torch.rand(size, device=self.device) * scale
        
        # 3. 计算三角函数
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t_plain for i in k])
        
        u_plain = torch.sin(tk)
        v_plain = torch.cos(tk)
        
        t2_plain = t_plain.square()
        t3_plain = t2_plain * t_plain

        # 4. 编码 (Float -> FixedPoint)
        t_enc = encoder.encode(t_plain, device=self.device).reshape(1, *size)
        u_enc = encoder.encode(u_plain, device=self.device)
        v_enc = encoder.encode(v_plain, device=self.device)
        t2_enc = encoder.encode(t2_plain, device=self.device).reshape(1, *size)
        t3_enc = encoder.encode(t3_plain, device=self.device).reshape(1, *size)
        
        # 5. 拼接
        results = torch.cat([t_enc, u_enc, v_enc, t2_enc, t3_enc], dim=0)
        
        # 6. 减去其他方的 Shares
        # 这一行必须存在！
        results -= self._get_additive_PRSS(results.shape, remove_rank=True)
        
        # 7. 转 CPU 返回
        return results.cpu()
    
    def generate_hybrid_triple_1(self, size, period, terms):
        """
        [服务端 - Debugged] 
        输出源头数据，确认 t 是浮点数且 sin/cos 计算正确。
        """
        encoder = FixedPointEncoder()
        
        # 1. 确定 Scaling
        if period > 50.0:
            scale = 100.0 
        else:
            scale = period
            
        # 2. 生成 t (明文浮点数)
        t_plain = torch.rand(size, device=self.device) * scale
        
        # 3. 计算三角函数
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t_plain for i in k])
        
        u_plain = torch.sin(tk)
        v_plain = torch.cos(tk)

        # 4. 编码 (Float -> FixedPoint)
        t_enc = encoder.encode(t_plain, device=self.device).reshape(1, *size)
        u_enc = encoder.encode(u_plain, device=self.device)
        v_enc = encoder.encode(v_plain, device=self.device)
        
        # 5. 拼接
        results = torch.cat([t_enc, u_enc, v_enc], dim=0)
        
        # 6. 减去其他方的 Shares
        # 这一行必须存在！
        results -= self._get_additive_PRSS(results.shape, remove_rank=True)
        
        # 7. 转 CPU 返回
        return results.cpu()

    def generate_cmp_aux(self, size):
        """
        [修复版] TFP Provider: 仅 Rank 0 生成数据，其他人为 0
        """
        from crypten.encoder import FixedPointEncoder
        import crypten.communicator as comm
        
        encoder = FixedPointEncoder()
        device = self.device
        my_rank = comm.get().get_rank() # 获取当前进程 ID
        
        # print(f"DEBUG: Rank {my_rank} generating CMP params...") 

        # ==================================================
        # 1. 只有 Rank 0 生成真值 (正数 a, r=1, c=0)
        #    其他 Rank 必须为 0，否则会破坏 Secret Sharing
        # ==================================================
        if my_rank == 0:
            # Rank 0: 生成合法的辅助参数
            a = torch.rand(size, device=device) * 10 + 100 # 保证是正数
            b = torch.rand(size, device=device) * (a * 1e-9)
            r = torch.ones(size, device=device)
            c = torch.zeros(size, device=device)
        else:
            # Rank 1+: 必须初始化为 0 (占位符)
            a = torch.zeros(size, device=device)
            b = torch.zeros(size, device=device)
            r = torch.zeros(size, device=device)
            c = torch.zeros(size, device=device)
        
        # 2. 编码并堆叠
        stacked = torch.stack([
            encoder.encode(a),
            encoder.encode(b),
            encoder.encode(r),
            encoder.encode(c)
        ])
        
        # 3. 转换为 Secret Shares
        # Rank 0: Secret - PRSS
        # Rank 1: 0 + PRSS (CrypTen 内部处理正负号)
        stacked -= self._get_additive_PRSS(stacked.shape, remove_rank=True)
        
        return stacked.cpu()
    
    def wraps(self, size):
        r = [generate_random_ring_element(size, generator=g) for g in self.generators]
        theta_r = count_wraps(r)

        return theta_r - self._get_additive_PRSS(size, remove_rank=True)

    def B2A(self, size):
        rB = self._get_binary_PRSS(size, bitlength=1)

        # Subtract all other shares of `rA` from plaintext value of `rA`
        rA = rB - self._get_additive_PRSS(size, remove_rank=True)

        return rA

    def generate_trig_triple(self, size, period, terms):
        """Generate trigonometric triple of given size"""
        encoder = FixedPointEncoder()
        t = torch.rand(size, device=self.device) * period
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t for i in k])
        u, v = torch.sin(tk), torch.cos(tk)
        t = encoder.encode(t, device=self.device).reshape(1, *t.shape)
        u = encoder.encode(u, device=self.device)
        v = encoder.encode(v, device=self.device)
        t -= self._get_additive_PRSS(t.shape, remove_rank=True)
        u -= self._get_additive_PRSS(u.shape, remove_rank=True)
        v -= self._get_additive_PRSS(v.shape, remove_rank=True)
        results = torch.cat([t.cpu(), u.cpu(), v.cpu()], dim=0)
        return results

    def generate_one_hot_pair(self, size, length):
        encoder = FixedPointEncoder(precision_bits=0)
        r = self._get_additive_PRSS(size) % length
        v = torch.nn.functional.one_hot(r, num_classes=length)
        v = encoder.encode(v, device=self.device)
        v -= self._get_additive_PRSS(v.shape, remove_rank=True)
        return v
