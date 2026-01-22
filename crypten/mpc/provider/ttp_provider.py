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
        a = generate_random_ring_element(size0, generator=generator, device=device)
        b = generate_random_ring_element(size1, generator=generator, device=device)
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

    def generate_hybrid_triple(self, size, period, terms, device=None):
        """
        Generate hybrid triples (Fourier + Cube).
        Returns: t, u, v, t2, t3
        """
        generator = TTPClient.get().get_generator(device=device)
        
        # 确定形状
        # t, t2, t3: size
        # u, v: (terms, *size)
        uv_shape = (terms,) + size
        
        if comm.get().get_rank() == 0:
            # TTP Server 返回 cat 后的张量
            # 结构: [t(1), u(terms), v(terms), t2(1), t3(1)] along dim 0
            cat_res = TTPClient.get().ttp_request("generate_hybrid_triple", device, size, period, terms)
            
            # 切片还原
            # t: index 0
            # u: index 1 ~ 1+terms
            # v: index 1+terms ~ 1+2*terms
            # t2: index 1+2*terms
            # t3: index 1+2*terms+1
            
            idx_u_start = 1
            idx_v_start = 1 + terms
            idx_t2 = 1 + 2 * terms
            idx_t3 = idx_t2 + 1
            
            t = cat_res[0]
            u = cat_res[idx_u_start:idx_v_start]
            v = cat_res[idx_v_start:idx_t2]
            t2 = cat_res[idx_t2]
            t3 = cat_res[idx_t3]
        else:
            t = generate_random_ring_element(size, generator=generator, device=device)
            u = generate_random_ring_element(uv_shape, generator=generator, device=device)
            v = generate_random_ring_element(uv_shape, generator=generator, device=device)
            t2 = generate_random_ring_element(size, generator=generator, device=device)
            t3 = generate_random_ring_element(size, generator=generator, device=device)

        return (
            ArithmeticSharedTensor.from_shares(t),
            ArithmeticSharedTensor.from_shares(u),
            ArithmeticSharedTensor.from_shares(v),
            ArithmeticSharedTensor.from_shares(t2),
            ArithmeticSharedTensor.from_shares(t3)
        )

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

    def generate_hybrid_triple(self, size, period, terms):
        """
        Generate hybrid triples (Fourier + Cube).
        Logic mirrors TrustedFirstParty.generate_hybrid_triple
        """
        encoder = FixedPointEncoder()
        
        # 1. 确定 Scaling
        if period > 50.0:
            scale = 100.0 
        else:
            scale = period
            
        # 2. 生成明文数据 (Floats)
        t_plain = torch.rand(size, device=self.device) * scale
        
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t_plain for i in k])
        
        u_plain = torch.sin(tk)
        v_plain = torch.cos(tk)
        t2_plain = t_plain.square()
        t3_plain = t2_plain * t_plain

        # 3. 编码 (Float -> FixedPoint Integer)
        # 注意: 为了能够 concat，需要统一维度
        # t, t2, t3 需要 unsqueeze 变成 (1, *size)
        
        t_enc = encoder.encode(t_plain, device=self.device).reshape(1, *size)
        u_enc = encoder.encode(u_plain, device=self.device) # (terms, *size)
        v_enc = encoder.encode(v_plain, device=self.device) # (terms, *size)
        t2_enc = encoder.encode(t2_plain, device=self.device).reshape(1, *size)
        t3_enc = encoder.encode(t3_plain, device=self.device).reshape(1, *size)
        
        # 4. 拼接
        results = torch.cat([t_enc, u_enc, v_enc, t2_enc, t3_enc], dim=0)
        
        # 5. 减去其他方的 shares
        results -= self._get_additive_PRSS(results.shape, remove_rank=True)
        
        # 必须移回 CPU 以便发送 (Crypten TTP 通信层通常处理 CPU tensor)
        return results.cpu()

    def generate_cmp_aux(self, size):
        """
        Generate CMP auxiliary parameters (a, b, r, c).
        """
        encoder = FixedPointEncoder()
        
        # 1. 生成明文逻辑 (同 TFP)
        a_abs = torch.rand(size, device=self.device) * 10 + 100 
        b_abs = torch.rand(size, device=self.device) * (a_abs * 1e-9) 
        
        sign = (torch.rand(size, device=self.device) > 0.5).float() * 2 - 1
        
        a = a_abs * sign
        b = b_abs * sign
        r = sign 
        c = (1 - sign) / 2
        
        # 2. 编码并堆叠
        # a, b, r, c 都是 Float tensor，编码为 Ring Element
        stacked = torch.stack([
            encoder.encode(a, device=self.device),
            encoder.encode(b, device=self.device),
            encoder.encode(r, device=self.device),
            encoder.encode(c, device=self.device)
        ])
        
        # 3. 减去其他方的 shares
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
