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

    def generate_3smp_triple(self, size, device=None):
        if comm.get().get_rank() == 0:
            a = generate_random_ring_element(size, device=device)
            b = generate_random_ring_element(size, device=device)
            c = generate_random_ring_element(size, device=device)

            ab = a.mul(b)
            bc = b.mul(c)
            ca = c.mul(a)
            abc = ab.mul(c)
        else:
            dummy = torch.empty(size, device=device)
            a = b = c = ab = bc = ca = abc = dummy

        results = [
            ArithmeticSharedTensor(x, precision=0, src=0) 
            for x in [a, b, c, ab, bc, ca, abc]
        ]
        
        return tuple(results)
    def square(self, size, device=None):
        """Generate square double of given size"""
        r = generate_random_ring_element(size, device=device)
        r2 = r.mul(r)

        # Stack to vectorize scatter function
        stacked = torch_stack([r, r2])
        stacked = ArithmeticSharedTensor(stacked, precision=0, src=0)
        return stacked[0], stacked[1]

    def cube_2(self, size, device=None, mode="cube"):
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
            l1 = generate_random_ring_element(size, device=device)%100
            l2 = generate_random_ring_element(size, device=device)%100
            
            l1_l2 = l1.mul(l2)
            l2_sq = l2.mul(l2)
            l1_l2_sq = l1.mul(l2_sq)

            l1 = ArithmeticSharedTensor(l1.float(), precision=0, src=0)
            l2 = ArithmeticSharedTensor(l2.float(), precision=0, src=0)
            l1_l2 = ArithmeticSharedTensor(l1_l2.float(), precision=8, src=0)
            l2_sq = ArithmeticSharedTensor(l2_sq.float(), precision=8, src=0)
            l1_l2_sq = ArithmeticSharedTensor(l1_l2_sq.float(), precision=8, src=0)
            
            return l1, l2, l1_l2, l2_sq, l1_l2_sq
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")
    def cube_1(self, size, device=None, mode="cube"):
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
            
            return stacked[0], stacked[1], stacked[2], stacked[3], stacked[4]
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")
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
            
            return stacked[0], stacked[1], stacked[2], stacked[3], stacked[4]
        
        else:
            raise ValueError(f"Unknown cube mode: {mode}")
        
    def generate_gelu_offline_batch(self, input_size, double_size, period, terms, device=None):
        from crypten.encoder import FixedPointEncoder
        encoder = FixedPointEncoder()
        
        num_double = double_size.numel() if isinstance(double_size, torch.Size) else torch.tensor(double_size).prod().item()
        num_input = input_size.numel() if isinstance(input_size, torch.Size) else torch.tensor(input_size).prod().item()
        
        len_cmp = 4 * num_double
        len_trip_d = 3 * num_double
        len_trip_i = 3 * num_input
        len_hybrid = (3 + 2 * terms) * num_input
        
        total_len = len_cmp + len_trip_d + len_hybrid + len_trip_i + len_trip_d
        
        if comm.get().get_rank() == 0:

            a = torch.rand(double_size, device=device) * 10 + 100 
            b = torch.rand(double_size, device=device) * (a * 1e-9)
            r = torch.ones(double_size, device=device)
            c = torch.zeros(double_size, device=device)
            raw_cmp = torch.cat([encoder.encode(x, device=device).reshape(-1) for x in [a,b,r,c]])
            
            def gen_triple_plain(sz):
                ta = torch.rand(sz, device=device)
                tb = torch.rand(sz, device=device)
                tc = ta * tb
                return torch.cat([encoder.encode(x, device=device).reshape(-1) for x in [ta, tb, tc]])
            
            raw_trip1 = gen_triple_plain(double_size)
            raw_trip2 = gen_triple_plain(input_size)
            raw_trip3 = gen_triple_plain(double_size)
            
            scale = 100.0 if period > 50 else period
            t = torch.rand(input_size, device=device) * scale
            k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
            tk = torch.stack([i * t for i in k])
            u, v = torch.sin(tk), torch.cos(tk)
            t2, t3 = t.square(), t.square() * t
            
            raw_hybrid = torch.cat([
                encoder.encode(t, device=device).reshape(-1),
                encoder.encode(u, device=device).reshape(-1),
                encoder.encode(v, device=device).reshape(-1),
                encoder.encode(t2, device=device).reshape(-1),
                encoder.encode(t3, device=device).reshape(-1)
            ])
            
            flat_data = torch.cat([raw_cmp, raw_trip1, raw_hybrid, raw_trip2, raw_trip3])
            
            assert flat_data.numel() == total_len, f"Size mismatch! Calc: {total_len}, Real: {flat_data.numel()}"
            
        else:

            flat_data = torch.empty(total_len, device=device)

        packed_ast = ArithmeticSharedTensor(flat_data, precision=0, src=0)
        
        cursor = 0
        def pull_ast(length):
            nonlocal cursor
            res = packed_ast[cursor : cursor+length]
            cursor += length
            return res
            
        res_cmp_chunk = pull_ast(4 * num_double).reshape(4, *double_size)
        res_cmp = tuple(res_cmp_chunk[i] for i in range(4))
        
        # Helper for Triples
        def unpack_triple(total_len, shape):
            chunk = pull_ast(total_len * 3).reshape(3, *shape)
            return (chunk[0], chunk[1], chunk[2])
            
        res_trip1 = unpack_triple(num_double, double_size)
        
        # Hybrid
        hybrid_len = (3 + 2*terms) * num_input
        hybrid_chunk = pull_ast(hybrid_len)
        chunk_sz = num_input
        h_t = hybrid_chunk[0:chunk_sz].reshape(input_size)
        h_u = hybrid_chunk[chunk_sz : chunk_sz*(1+terms)].reshape(terms, *input_size)
        h_v = hybrid_chunk[chunk_sz*(1+terms) : chunk_sz*(1+2*terms)].reshape(terms, *input_size)
        h_t2 = hybrid_chunk[chunk_sz*(1+2*terms) : chunk_sz*(2+2*terms)].reshape(input_size)
        h_t3 = hybrid_chunk[chunk_sz*(2+2*terms) : ].reshape(input_size)
        res_hybrid = (h_t, h_u, h_v, h_t2, h_t3)
        
        res_trip2 = unpack_triple(num_input, input_size)
        res_trip3 = unpack_triple(num_double, double_size)
        
        return res_cmp, res_trip1, res_hybrid, res_trip2, res_trip3
    
    def generate_hybrid_triple(self, size, period, terms, device=None):
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
    
    def generate_hybrid_triple_1(self, size, period, terms, device=None):
        if period > 50.0:
            scale = 100.0 
        else:
            scale = period
            
        t_plain = torch.rand(size, device=device) * scale
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        tk = torch.stack([i * t_plain for i in k])
        u_plain, v_plain = torch.sin(tk), torch.cos(tk)

        t = ArithmeticSharedTensor(t_plain, src=0)
        u = ArithmeticSharedTensor(u_plain, src=0)
        v = ArithmeticSharedTensor(v_plain, src=0)

        return t, u, v

    def generate_cmp_aux(self, size, device=None):
        a_abs = torch.rand(size, device=device) * 10 + 1 
        
        b_abs = torch.rand(size, device=device) * (a_abs * 1e-9) 
        
        sign = (torch.rand(size, device=device) > 0.5).float() * 2 - 1
        
        a = a_abs * sign
        b = b_abs * sign

        r = sign 
        c = (1 - sign) / 2
        
        enc_a = ArithmeticSharedTensor(a, precision=0, src=0)
        enc_b = ArithmeticSharedTensor(b, src=0)
        enc_r = ArithmeticSharedTensor(r, src=0)
        enc_c = ArithmeticSharedTensor(c, src=0)
        
        return enc_a, enc_b, enc_r, enc_c
    
    def generate_cmp_aux(self, size, device=None):
        a_abs = torch.rand(size, device=device) * 10 + 1 
        
        b_abs = torch.rand(size, device=device) * (a_abs * 1e-9) 
        
        sign = (torch.rand(size, device=device) > 0.5).float() * 2 - 1
        
        a = a_abs * sign
        b = b_abs * sign

        r = sign 
        c = (1 - sign) / 2
        
        enc_a = ArithmeticSharedTensor(a,  src=0)
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
    