#!/usr/bin/env python3

# Modified by Andes Y. L. Kei: Implemented alternative approximations for Sigmoid, Tanh, Erf, GELU, and Softmax
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import crypten
import torch
from crypten.config import cfg
import numpy as np
import scipy.special
from scipy.linalg import lstsq

SIGMOID_K1 = 1  
SIGMOID_K2 = 4  
ERF_K1 = 1  
ERF_K2 = 4 
TANH_K1 = 1  
TANH_K2 = 9  
INVSQRT_K1 = 1  
INVSQRT_K2 = 12  
SILU_K1 = 1
SILU_K2 = 5
INV_K1 = 1
INV_K2 = 9
GELU_K1 = 1
GELU_K2 = 9
EXP_K1 = 1
EXP_K2 = 9
_SIGMOID_PARAMS_TABLE = {
   
    (1, 8): {
        "a0": 0.4999999999999999,
        "poly_body": [0.062460999507739506],
        "beta_sin_coeffs": [0.25008196918498904, 0.06702957954702526, 0.019433109276880533, 0.005637113032415615, 0.00165073835817726, 0.0004760697094664209, 0.0001409357228862451, 4.002775103285762e-05]
    },
    (1, 12): {
        "a0": 0.49999999999999994,
        "poly_body": [0.06245865602295103],
        "beta_sin_coeffs": [0.25009390445491203, 0.06702361191221094, 0.019437087699926418, 0.005634129215302933, 0.0016531254116904826, 0.0004740804983852596, 0.00014264076077355283, 3.853584306557718e-05, 1.3261692657364522e-05, 2.4387471242228864e-06, 1.7235102838867135e-06, -2.2987037912434103e-07]
    },

    (2, 12): {
        "a0": 0.49999999999999994,
        "poly_body": [0.06245865602295103, 0.0],
        "beta_sin_coeffs": [0.25009390445491203, 0.06702361191221094, 0.019437087699926418, 0.005634129215302933, 0.0016531254116904826, 0.0004740804983852596, 0.00014264076077355283, 3.853584306557718e-05, 1.3261692657364522e-05, 2.4387471242228864e-06, 1.7235102838867135e-06, -2.2987037912434103e-07]
    },

    (3, 12): {
        "a0": 0.49999999999999967,
        "poly_body": [0.06291245771807705, 0.0, -7.099806574617745e-06],
        "beta_sin_coeffs": [0.24869003754238217, 0.06719797624035966, 0.019385976805583065, 0.005655365238431782, 0.0016424674226812952, 0.00048009634832841283, 0.00013896545357787782, 4.091058714992843e-05, 1.1663421873592467e-05, 3.5471887878075966e-06, 9.378034743379018e-07, 3.356051459000025e-07]
    }
}

_TANH_PARAMS_TABLE = {

    (1, 8): {
        "a0": 1.4915878357495863e-16,
        "poly_body": [0.16677790053468844],
        "beta_sin_coeffs": [
            0.5697071168252006, 0.21017168288239746, 0.08930825758091233, 
            0.039178767726712964, 0.017060735094000988, 0.0076034536718419645, 
            0.0032473342091208957, 0.0015076450980772605
        ]
    },
    (1, 12): {
        "a0": -2.5906525568282286e-17,
        "poly_body": [0.14287078333242556],
        "beta_sin_coeffs": [0.5867321860945147, 0.2330779497138758, 0.10986841030369815, 0.053714360179613695, 0.026450300784987585, 0.013077423011490693, 0.0064469618961475105, 0.00319750781235197, 0.001569292685021838, 0.0007849720659307415, 0.0003792161979511833, 0.00019529371770981205]
    },

    (2, 12): {
        "a0": 1.4915878357495863e-16,
        "poly_body": [0.1666689243771898, 0.0],
        "beta_sin_coeffs": [
            0.5701233750812582, 0.20996355375950404, 0.08944701032380079, 
            0.039074703175538295, 0.017143986728777396, 0.007534077315805268, 
            0.003306799650793567, 0.0014556128430336219, 0.0006369990937402685, 
            0.00028215285531896006, 0.00012186538199317154, 5.5474985829787186e-05
        ]
    },

    (3, 12): {
        "a0": 1.491587835749586e-16,
        "poly_body": [0.1745374246430353, 0.0, -0.00021885165768138846],
        "beta_sin_coeffs": [
            0.5518670629743909, 0.21223104048960384, 0.08878234871601902, 
            0.03935086287606917, 0.01700538699966389, 0.0076123092572879265, 
            0.003259004838726005, 0.0014864947370296405, 0.0006162146945934667, 
            0.00029656736767917836, 0.00011164781178621178, 6.28286014628608e-05
        ]
    }
}

_INVSQRT_PARAMS_TABLE = {
    # === K1 = 1 (线性辅助) ===
    # 小范围段 (0.1 ~ 8.0)
    (1, 8, 8.0): {
        "a0": 2.673844851983361,
        "poly_body": [-0.2923985979672142],
        "beta_sin_coeffs": [-1.20207164876241, -0.5032847916291682, -0.2775955734241775, -0.1812597365906886, -0.1181687789471339, -0.08538400220005772, -0.056445277899837915, -0.037925740873904316]
    },
    # 大范围段 (8.0 ~ 2048.0)
    (1, 8, 2048.0): {
        "a0": 0.20192735731192768,
        "poly_body": [-8.901384327141743e-05],
        "beta_sin_coeffs": [-0.0964281445416358, -0.042859424938057625, -0.02433259508566518, -0.01689475521421311, -0.011426572684851614, -0.008872906179404502, -0.006210026103745888, -0.004604462117425797]
    },
    # 小范围段 (k2=12)
    (1, 12, 8.0): {
        "a0": 3.0171173347846336,
        "poly_body": [-0.33432127358922564],
        "beta_sin_coeffs": [-1.4255494043797743, -0.6099011473575614, -0.3519140211582302, -0.2344610912939924, -0.16273906356690698, -0.12121455071475327, -0.08924628505593063, -0.06953856092260546, -0.05196210312485785, -0.040953431422271674, -0.02974313616068201, -0.02083495076259728]
    },
    # 大范围段 (k2=12)
    (1, 12, 2048.0): {
        "a0": 0.23229589487335833,
        "poly_body": [-0.00010346435572836789],
        "beta_sin_coeffs": [-0.1162581028408835, -0.05228999970035461, -0.0309572108364939, -0.02163608673648766, -0.015438322021931204, -0.012114045824321431, -0.009219052795826355, -0.007621610586266408, -0.005945454378107071, -0.005026808618778701, -0.003888197777505186, -0.0029994497011014737]
    },

    # === K1 = 2 (线性+二次项辅助) ===
    (2, 12, 8.0): {
        "a0": 3.0255212131680485,
        "poly_body": [-1.4132534769396887, 0.1347943359381504],
        "beta_sin_coeffs": [0.7925148345780794, -0.6113691481051957, -0.2720558382324872, -0.23518813237886463, -0.1464728085118898, -0.12169263361022133, -0.08384983403743268, -0.06989201354058397, -0.049752944683980595, -0.041238083781684004, -0.028765822794154173, -0.021146184384216563]
    },
    (2, 12, 2048.0): {
        "a0": 0.25519088779093274,
        "poly_body": [-0.01810278687662938, 8.787075499809483e-06],
        "beta_sin_coeffs": [9.36822801830582, -0.054501905012029915, 0.3129939658454932, -0.02274398363764281, 0.05569033905767392, -0.012857452823997539, 0.014970190655182286, -0.008192069265056752, 0.004339382936880741, -0.005528119004682493, 0.0009419902333236877, -0.0037978430096622913]
    },

    # === K1 = 3 (更高阶，稳定性较差) ===
    (3, 12, 8.0): {
        "a0": 3.0484134131894134,
        "poly_body": [-2.689768910416188, 0.6659601797212282, -0.04652397491174194],
        "beta_sin_coeffs": [0.33957092128042815, 0.5289714950472427, -0.29032599430567846, -0.09713377452156494, -0.15104885320531147, -0.08298620565811496, -0.085848748941496, -0.0548446116463774, -0.05089250856006637, -0.034357433377043854, -0.02952220874957167, -0.017636494205118203]
    },
    (3, 12, 2048.0): {
        "a0": 0.305799310532323,
        "poly_body": [-0.042196543328671426, 3.155056794567181e-05, -5.378313820213747e-09],
        "beta_sin_coeffs": [16.10027928905099, 2.1593738901504476, 0.5557562166063786, 0.24606997022405447, 0.10528569900318176, 0.06287862445544036, 0.03148454138184424, 0.021440871948683386, 0.01111134382422052, 0.008089505149290174, 0.0038922626738960996, 0.0027902008572456127]
    }
}

_RECIPROCAL_PARAMS_TABLE = {
    
    # === K1 = 1 (线性辅助) ===
    # 小范围段 (0.1 ~ 8.0)
    (1, 8, 8.0): {
        "a0": 5.0222197838772225,
        "poly_body": [-0.6141288317888934],
        "beta_sin_coeffs": [-2.811344541718986, -1.2243676039256202, -0.6925980445056017, -0.4381699794077579, -0.28064254844181447, -0.18283548009672626, -0.10772618319849936, -0.051778520638824245]
    },
    # 大范围段 (8.0 ~ 2048.0)
    (1, 8, 2048.0): {
        "a0": 0.02339405716011906,
        "poly_body": [-1.1232950093836347e-05],
        "beta_sin_coeffs": [-0.013343155046296663, -0.005958958453677971, -0.0034417234583903225, -0.002240838265044309, -0.001472611315327415, -0.0009938471925751465, -0.0006065470354893117, -0.000306608613854517]
    },
    # 小范围段 (k2=12)
    (1, 12, 8.0): {
        "a0": 6.722286652142002,
        "poly_body": [-0.8259223950928658],
        "beta_sin_coeffs": [-3.896696673605398, -1.7624134815568546, -1.0528039212596791, -0.7059082469821661, -0.49607825591949667, -0.3620866708458491, -0.26466787831259375, -0.19586038500758993, -0.1405884605006413, -0.09904906022447335, -0.06268740498315341, -0.03123447713684064]
    },
    # 大范围段 (k2=12)
    (1, 12, 2048.0): {
        "a0": 0.03302602671784331,
        "poly_body": [-1.5924055910611643e-05],
        "beta_sin_coeffs": [-0.019490238808237022, -0.009016691357671247, -0.005491318700829544, -0.003773285589159699, -0.002710697625080689, -0.0020337689690383544, -0.001524831626149393, -0.0011647456873814259, -0.0008617881112036451, -0.0006307408430033456, -0.00041539398932058076, -0.00021811571235416237]
    },

    # === K1 = 2 (线性+二次项辅助) ===
    (2, 12, 8.0): {
        "a0": 6.768001232054899,
        "poly_body": [-3.7711934849517297, 0.36758657283045937],
        "beta_sin_coeffs": [2.138412255101094, -1.7740272758772009, -0.8395477164094776, -0.71164047728967, -0.45437006911482153, -0.3658309269147162, -0.2517846100088192, -0.19860301071877431, -0.13594162890816297, -0.10122239823882302, -0.06113521487464308, -0.0332571362793708]
    },
    (2, 12, 2048.0): {
        "a0": 0.03448307226954854,
        "poly_body": [-0.0005050622210036153, 2.385820796056008e-07],
        "beta_sin_coeffs": [0.23752455495282063, -0.009356956261228393, 0.0036799299770615915, -0.003943455688954444, -0.0008799933293302975, -0.002147651313730391, -0.000940340754798533, -0.001251552425679986, -0.0006402301100078829, -0.0007041999956354765, -0.00033534533241263585, -0.00029549656983883347]
    },

    # === K1 = 3 (线性+二次+三次项辅助) ===
    (3, 12, 8.0): {
        "a0": 6.769748434941708,
        "poly_body": [-3.8966804423070274, 0.4100595636103737, -0.0033535929226171973],
        "beta_sin_coeffs": [2.174705307256502, -1.6918084961650117, -0.8383588756470174, -0.7016789025714697, -0.45417875232858196, -0.3630331358150978, -0.25174891003202277, -0.19751081165197917, -0.1359438664626984, -0.1007162791225666, -0.061144643508824044, -0.03298272670727847]
    },
    (3, 12, 2048.0): {
        "a0": 0.05993343942424357,
        "poly_body": [-0.008777866134197562, 6.452532508389537e-06, -1.0647743616685983e-09],
        "beta_sin_coeffs": [3.4063981272693247, 0.4248928767376657, 0.11632230493655041, 0.047254845648278206, 0.021410505691275235, 0.01149793148942666, 0.006062841128624675, 0.003595983850539763, 0.0019372278047675487, 0.001155004907629837, 0.0005451184970294063, 0.00026205597848153725]
    }
}

_EXP_PARAMS_TABLE = {

    # --- K1 = 1 (线性项辅助) ---
    (1, 8): {
        "a0": 0.21747074739249442,
        "poly_body": [0.013591914871886854],
        "beta_sin_coeffs": [
            0.12489855543407198, 0.04605954938088917, 0.018666085067584978, 
            0.007014391596379622, 0.002273000886161232, 0.0005889024439924406, 
            0.0001074794945164943, 1.0220288678319571e-05
        ]
    },
    (1, 12): {
        "a0": 0.3253291232344465,
        "poly_body": [0.0203330632631962],
        "beta_sin_coeffs": [
            0.19038760702797833, 0.07441333557710686, 0.03343608087861552, 
            0.014735842405395785, 0.006039852045461626, 0.0022090490270534626, 
            0.0006783203654527648, 0.00015229357392128082, 1.2311942943631288e-05, 
            -7.735925260867457e-06, -3.8366920692307995e-06, -6.777548826102553e-07
        ]
    },

    # --- K1 = 2 (线性+二次项辅助) ---
    (2, 12): {
        "a0": 0.3829918766227242,
        "poly_body": [0.023937871312459467, 5.537858077592215e-08],
        "beta_sin_coeffs": [
            0.2258444787901797, 0.09038309551497638, 0.04235650867791144, 
            0.01992903078131213, 0.009007978190754181, 0.0038199013978471082, 
            0.0014863226291764917, 0.0005160767771325118, 0.00015363864481127644, 
            3.679891346553182e-05, 6.306330911118898e-06, 5.805678292463508e-07
        ]
    },

    # --- K1 = 3 (更高阶项) ---
    (3, 12): {
        "a0": 0.5403991327700272,
        "poly_body": [0.12198698859987614, 0.008269834013038857, 0.00017228630599438143],
        "beta_sin_coeffs": [0.04994447007497581, 0.10060016014332454, 0.05758096785303508, 0.030911762147992135, 0.015954776621073558, 0.007844942101016814, 0.0036071974367934416, 0.0015079553546010424, 0.0005490296123522897, 0.00016248464707917452, 3.449558986067898e-05, 3.9063805629131515e-06]
    }
}

_ERF_PARAMS_TABLE = {

    (1, 8): {
        "a0": 1.3110272030009522e-16,
        "poly_body": [0.20000627904783783],
        "beta_sin_coeffs": [
            0.5767690216073977, 0.21449529610391022, 0.08728857355067796, 
            0.03281526448432516, 0.010793707088840478, 0.0030418054544940265, 
            0.000719062731070722, 0.00014622877714586186
        ]
    },
    (1, 12): {
        "a0": 1.3110272030009522e-16,
        "poly_body": [0.20000000079322103],
        "beta_sin_coeffs": [
            0.5767890059123588, 0.2144853039516764, 0.08729523498522647, 
            0.03281026840870141, 0.010797703949043698, 0.0030384747379595687, 
            0.000721917630651953, 0.00014373074032003716, 2.386258981330265e-05, 
            3.2930528187513254e-06, 3.7651467064182807e-07, 3.5888860945881957e-08
        ]
    },

    (2, 12): {
        "a0": 1.3110272030009522e-16,
        "poly_body": [0.20000000079322103, 0.0],
        "beta_sin_coeffs": [
            0.5767890059123588, 0.2144853039516764, 0.08729523498522647, 
            0.03281026840870141, 0.010797703949043698, 0.0030384747379595687, 
            0.000721917630651953, 0.00014373074032003716, 2.386258981330265e-05, 
            3.2930528187513254e-06, 3.7651467064182807e-07, 3.5888860945881957e-08
        ]
    },

    (3, 12): {
        "a0": 1.0875174062793401e-17,
        "poly_body": [0.20000138967986578, 0.0, -5.5627196253608204e-08],
        "beta_sin_coeffs": [
            0.5767863205268329, 0.2144856374843209, 0.08729513721779873, 
            0.03281030903000893, 0.010797683561918901, 0.0030384862453727663, 
            0.0007219106003438512, 0.00014373528284741365, 2.385953256216225e-05, 
            3.2951731005662933e-06, 3.750117318832469e-07, 3.697053009855227e-08
        ]
    }
}
__all__ = [
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "odrelu",
    "od_sign",
]

# Iterative methods:
def exp(self, k1=3, k2=12,fit_min=-16,fit_max=-2):
    r"""Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.作为 Gramine-TDX 的介绍 (基于 CCS '24 论文)
    """  # noqa: W605
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.exp_method
    iters = cfg.functions.exp_iterations

    if method == "ideal":
        return crypten.cryptensor(torch.exp(self.get_plain_text()), device=self.device)
    elif method == "limit":
        result = 1 + self.div(2**iters)
        for _ in range(iters):
            result = result.square()
        return result
    elif method == "newer":
        upper = -4.0
        lower = -16.0
        
        diffs = crypten.cat([self - upper, lower - self]).relu().split(self.shape[0])
        
        safe_x = self + diffs[1] - diffs[0]

        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2
        

        params = _EXP_PARAMS_TABLE[(use_k1, use_k2)]
        poly_coeffs = params["poly"]
        beta_sin_coeffs = params["beta"]
        period = 16.0
        
        poly_bias = poly_coeffs[0]
        poly_part = safe_x.polynomial(poly_coeffs[1:]) + poly_bias

        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        fourier_part = _fourier_series_x3(safe_x, len(beta_sin_coeffs), period, beta_sin=beta_sin)

        res = poly_part + fourier_part
        
        return res
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2

        L_fit = 16.0 
        # f_min = fit_min if fit_min is not None else -16
        f_max = fit_max if fit_max is not None else -2
        key = (use_k1, use_k2)
        # 1. 现场计算拟合系数
        if key in _EXP_PARAMS_TABLE:
            params = _EXP_PARAMS_TABLE[key]
            a0_val = params["a0"]
            poly_vals = params["poly_body"]
            beta_vals = params["beta_sin_coeffs"]
            
        else:
            a0_val, poly_vals, beta_vals= _get_dynamic_params("exp",use_k1, use_k2, fit_min, fit_max, f_max)

        # 2. 范围控制 (Range Reduction)
        full_poly = [0.0] + poly_vals
        # 3. 计算多项式部分
        poly_part =  self.mul(full_poly[1])
        # 4. 计算傅里叶部分
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_vals, device=self.device, dtype=torch.float)
        y_final = _fourier_series_x2x3_time(
            self, 
            len(beta_vals), 
            period, 
            alpha=a0_val,          # 传入常数项
            beta_sin=beta_sin, 
            poly_coeffs=full_poly  # 【关键】传入多项式系数
        )
        # 5. 合成
        y_final =poly_part+y_final 
        return y_final
    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2
        f_max = fit_max if fit_max is not None else -2
        L_fit = 16.0 
        key = (use_k1, use_k2)
        if key in _EXP_PARAMS_TABLE:
            params = _EXP_PARAMS_TABLE[key]
            a0_val = params["a0"]
            poly_vals = params["poly_body"]
            beta_vals = params["beta_sin_coeffs"]
            
        else:
            a0_val, poly_vals, beta_vals= _get_dynamic_params("exp",use_k1, use_k2, fit_min, fit_max, f_max)
        full_poly = [0.0] + poly_vals
        # 3. 计算多项式部分
        poly_part =  self.mul(full_poly[1])
        # 4. 计算傅里叶部分
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_vals, device=self.device, dtype=torch.float)
        y_final = _fourier_series_x2x3(
            self, 
            len(beta_vals), 
            period, 
            alpha=a0_val,          # 传入常数项
            beta_sin=beta_sin, 
            poly_coeffs=full_poly  # 【关键】传入多项式系数
        )
        # 5. 合成
        y_final =poly_part+y_final 
        return y_final
    elif method == "newer_time+":
        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12

        threshold = 8.0 
        L_small = threshold
        L_large = 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        key_s = (use_k1, use_k2)
        key_l = (use_k1, use_k2)
        if key_s in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key_s]
            a0_s = params["a0"]
            poly_s = params["poly_body"]
            beta_s = params["beta_sin_coeffs"]
            beta_s= torch.tensor(beta_s, device=device, dtype=torch.float)
            
        else:
            a0_s, poly_s, beta_s= _get_dynamic_params_1("exp", use_k1, use_k2, L=L_small, min_val=0.1)
            print(use_k1, use_k2, L_small)

        if key_l in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key_l]
            a0_l = params["a0"]
            poly_l = params["poly_body"]
            beta_l = params["beta_sin_coeffs"]
            beta_l= torch.tensor(beta_l, device=device, dtype=torch.float)
            
        else:
            a0_l, poly_l, beta_l= _get_dynamic_params_1("exp", use_k1, use_k2, L=L_large, min_val=L_small)

        period_s = 2 * L_small
        period_l = 2 * L_large

        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)
        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)[0]
        is_small = 1.0 - is_large
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)
        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l
        def _compute_hybrid_branch(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device) # sin(kd)
                q = torch.cos(delta_k).to(device) # cos(kd)
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            # 设置缩放因子防止溢出
            if i_flag == 0.0:
                s_sq = 32.0  
                s_cu = 32.0
            else:           
                s_sq = 256.0
                s_cu = 1024.0

            s_cu_pow3 = s_cu * s_cu * s_cu 


            # --- 计算 x^3 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                # 变量缩放
                d_s3  = delta_raw / s_cu
                t_s3  = t / s_cu
                t2_s3 = t2 / (s_cu * s_cu)
                t3_s3 = t3 / (s_cu * s_cu * s_cu)
                d_s3_sq = d_s3*d_s3
                
                cu_term1 = d_s3 * d_s3_sq
                cu_term2 = t_s3 * (3 * d_s3_sq) * -1.0
                cu_term3 = t2_s3 * (3 * d_s3)
                x_cu_scaled = cu_term1 + cu_term2 + cu_term3 - t3_s3
                
                # 融合乘法
                combined_c3 = c3 * s_cu_pow3
                poly_val += x_cu_scaled * combined_c3

            total = poly_val + fourier_val
            return total
        y_small = _compute_hybrid_branch(
            delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        y_large = _compute_hybrid_branch(
            delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        y0 = is_large
        a, b, c, ab, bc, ca, abc = provider.generate_3smp_triple(self.size(), device=self.device)
        t1=self+is_large-a-self
        t2=self+1-y_small-b-self
        t3=self+y_large-c-self
        stacked_shares = crypten.stack([t1, t2, t3], dim=0)
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
        eps = stacked_plain[0]
        rho = stacked_plain[1]
        sigma = stacked_plain[2]

        term_abc = abc
        term_2prod = (sigma * ab) + (rho * ca) + (eps * bc)
        term_1prod = (rho * sigma * a) + (eps * sigma * b) + (eps * rho * c)
        
        term_const = eps * rho * sigma
        
        # 结果 = abc + ...
        result = term_abc + term_2prod + term_1prod + term_const
        
        return self+y0+result-self
    else:
        raise ValueError(f"Invalid method {method} given for exp function")

def log(self, input_in_01=False):
    r"""
    Approximates the natural logarithm using 8th order modified
    Householder iterations. This approximation is accurate within 2% relative
    error on [0.0001, 250].

    Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

    .. math::

        y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the domain [0, 1],
            causing the function optimize for this domain. This is useful for computing
            log-probabilities for entropy functions.

            We shift the domain of convergence by a constant :math:`a` using the following identity:

            .. math::

                \ln{u} = \ln {au} - \ln{a}

            Since the domain of convergence for CrypTen's log() function is approximately [1e-4, 1e2],
            we can set :math:`a=100`.

    Configuration parameters:
        iterations (int): number of Householder iterations for the approximation
        exp_iterations (int): number of iterations for limit approximation of exp
        order (int): number of polynomial terms used (order of Householder approx)
    """
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    iterations = cfg.functions.log_iterations
    exp_iterations = cfg.functions.log_exp_iterations
    order = cfg.functions.log_order

    term1 = self.div(120)
    term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
    y = term1 - term2 + 3.0

    # 8th order Householder iterations
    with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
        for _ in range(iterations):
            h = 1 - self * exp(-y)
            y -= h.polynomial([1 / (i + 1) for i in range(order)])
    return y


def reciprocal(self, input_in_01=False, k1=3, k2=12, L=2048.0):
    r"""
    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

    Configuration params:
        reciprocal_method (str):  One of 'NR' or 'log'.
        reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method
        reciprocal_log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method
        reciprocal_all_pos (bool): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input.
        reciprocal_initial (tensor): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    pos_override = {"functions.reciprocal_all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64),k1=k1,k2=k2,L=L).mul(64)
        return rec

    method = cfg.functions.reciprocal_method
    all_pos = cfg.functions.reciprocal_all_pos
    initial = cfg.functions.reciprocal_initial

    if method == "ideal":
        return crypten.cryptensor(torch.reciprocal(self.get_plain_text()), device=self.device)

    if not all_pos:
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos,k1=k1,k2=k2,L=L)

    if method == "NR":
        nr_iters = cfg.functions.reciprocal_nr_iters
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result += result - result.square().mul_(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal_log_iters
        with cfg.temp_override({"functions.log_iters": log_iters}):
            return exp(-log(self))
    elif method == "newer":
        use_k1 = k1 if k1 is not None else INV_K1
        use_k2 = k2 if k2 is not None else INV_K2
        i_sqrt = self.inv_sqrt(k1=use_k1, k2=use_k2)
        return i_sqrt.square()
    elif method == "newer_1":
        use_k1 = k1 if k1 is not None else INV_K1 
        use_k2 = k2 if k2 is not None else INV_K2
        
        threshold = 2.0
        L_small = threshold
        current_L = 32.0 
        L_large = current_L if current_L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        # --- Group A: 小范围 [0.1, 2.0] ---
        a0_s, poly_s, beta_s = _get_dynamic_params("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)
        period_s = 2 * L_small
        t_s, u_s, v_s = provider.generate_trig_triple(self.size(), period_s, len(beta_s), device=device)

        # --- Group B: 大范围 [2.0, 32.0] ---
        a0_l, poly_l, beta_l = _get_dynamic_params("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)
        period_l = 2 * L_large
        t_l, u_l, v_l = provider.generate_trig_triple(self.size(), period_l, len(beta_l), device=device)

        # 生成掩码
        is_large = (self - threshold).od_sign()[0]
        is_small = 1.0 - is_large

        # 计算两段的 delta
        delta_s_share = self - t_s + period_s
        delta_l_share = self - t_l + period_l
        
        # 优化通信：打包 reveal
        stacked_shares = crypten.stack([delta_s_share, delta_l_share], dim=0)
        stacked_plain = stacked_shares.get_plain_text()
        
        delta_s_plain = stacked_plain[0]
        delta_l_plain = stacked_plain[1]

        # 内部函数：计算 Fourier + Polynomial
        # 这部分逻辑和 inv_sqrt 完全一样，可以直接复用
        def _compute_local_fourier(delta_p, period, beta_list, u_share, v_share, poly_coeffs, a0_val):
            delta = torch.remainder(delta_p, period)
            k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
            
            delta_k = torch.stack([i * delta for i in k_list])
            p = torch.sin(delta_k).to(device)
            q = torch.cos(delta_k).to(device)
            
            beta_tensor = torch.tensor(beta_list, device=device)
            view_shape = [-1] + [1] * self.dim()
            beta_tensor = beta_tensor.view(view_shape)

            fourier_val = ((v_share * p + u_share * q) * beta_tensor).sum(dim=0)
            poly_val = self.polynomial(poly_coeffs) + a0_val
            
            return poly_val + fourier_val

        # 计算初始猜测值 y0
        y_small = _compute_local_fourier(delta_s_plain, period_s, beta_s, u_s, v_s, poly_s, a0_s)
        y_large = _compute_local_fourier(delta_l_plain, period_l, beta_l, u_l, v_l, poly_l, a0_l)

        values = crypten.stack([y_small, y_large], dim=0)

        masks = crypten.stack([is_small, is_large], dim=0)
        products = values.mul(masks)
        y0 = products.sum(dim=0)
        
        y0_sq = y0.square()
        term1 = self.mul(y0_sq) # x * y0^2
        y1 = y0.mul(2.0) - term1

        y1_sq = y1.square()
        term2 = self.mul(y1_sq)
        y2 = y1.mul(2.0) - term2

        
        return y2
    elif method == "newer_time":
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 1 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 2048.0 
        key_s = (use_k1, use_k2, L_small)
        key_l = (use_k1, use_k2, L_large)
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        # 2. 获取拟合参数 (Small & Large)
        # a0_s, poly_s, beta_s = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)
        # a0_l, poly_l, beta_l = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)
        if key_s in _RECIPROCAL_PARAMS_TABLE:
            params = _RECIPROCAL_PARAMS_TABLE[key_s]
            a0_s = params["a0"]
            poly_s = params["poly_body"]
            beta_s = params["beta_sin_coeffs"]
            beta_s= torch.tensor(beta_s, device=device, dtype=torch.float)
            
        else:
            a0_s, poly_s, beta_s= _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)

        if key_l in _RECIPROCAL_PARAMS_TABLE:
            params = _RECIPROCAL_PARAMS_TABLE[key_l]
            a0_l = params["a0"]
            poly_l = params["poly_body"]
            beta_l = params["beta_sin_coeffs"]
            beta_l= torch.tensor(beta_l, device=device, dtype=torch.float)
            
        else:
            a0_l, poly_l, beta_l= _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)

        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)
        # 4. 计算分段掩码 (CMP Protocol)
        diff = self - threshold
        u_cmp, v_cmp, w_cmp= provider.generate_additive_triple(self.size(),self.size(),"mul",device=device)
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        eps_cmp_share = diff - u_cmp
        delta_cmp_share =self+cmp_a - v_cmp-self     
        delta_s_share = (self + t_s + period_s)
        delta_l_share = (self + t_l + period_l)
        # masked_sign_share = diff.mul(cmp_a).add(cmp_b)
        comm_block_1 = crypten.stack([
            eps_cmp_share, 
            delta_cmp_share,             
            delta_s_share,
            delta_l_share
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
        
        eps_cmp = plain_block_1[0]  
        delta_cmp = plain_block_1[1]  
        delta_s_plain_raw = plain_block_1[2]
        delta_l_plain_raw = plain_block_1[3]
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)    
        masked_share = (z_cmp + cmp_b)
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        # 7. 定义混合计算分支函数
        def _compute_hybrid_branch(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
            delta_k = torch.stack([i * delta_mod for i in k_list])
            p = torch.sin(delta_k).to(device) # sin(kd)
            q = torch.cos(delta_k).to(device) # cos(kd)
            if isinstance(beta_list, list):
                beta_list = torch.tensor(beta_list, device=self.device)
            beta_view = beta_list.view([-1] + [1] * self.dim())
            fourier_val = ((v * p - u * q) * beta_view).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            # 设置缩放因子防止溢出
            if i_flag == 0.0:
                s_sq = 32.0  
                s_cu = 32.0
            else:           
                s_sq = 256.0
                s_cu = 1024.0

            s_sq_pow2 = s_sq * s_sq
            s_cu_pow3 = s_cu * s_cu * s_cu 

            # --- 计算 x^2 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]
                
                # 变量缩放
                d_s2  = delta_raw / s_sq
                t_s2  = t / s_sq
                t2_s2 = t2 / s_sq_pow2
                d_s2_sq = d_s2.square()
                
                # x^2 = (d - t)^2 = d^2 - 2dt + t^2
                sq_term1 = d_s2_sq
                sq_term2 = t_s2 * (2 * d_s2) * -1.0
                sq_term3 = t2_s2
                x_sq_scaled = sq_term1 + sq_term2 + sq_term3
                
                # 融合乘法
                combined_c2 = c2 * s_sq_pow2
                poly_val += x_sq_scaled * combined_c2

            # --- 计算 x^3 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                # 变量缩放
                d_s3  = delta_raw / s_cu
                t_s3  = t / s_cu
                t2_s3 = t2 / (s_cu * s_cu)
                t3_s3 = t3 / (s_cu * s_cu * s_cu)
                d_s3_sq = d_s3.square()
                
                cu_term1 = d_s3 * d_s3_sq
                cu_term2 = t_s3 * (3 * d_s3_sq) * -1.0
                cu_term3 = t2_s3 * (3 * d_s3)
                x_cu_scaled = cu_term1 + cu_term2 + cu_term3 - t3_s3
                
                # 融合乘法
                combined_c3 = c3 * s_cu_pow3
                poly_val += x_cu_scaled * combined_c3

            total = poly_val + fourier_val
            return total

        y_small = _compute_hybrid_branch(
            delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        y_large = _compute_hybrid_branch(
            delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )
        y_curr = crypten.stack([y_small, y_large], dim=0)
        x_curr = crypten.stack([self, self], dim=0)
        target_device = self.device
        def to_raw_gpu(t):
            raw = t
            while not isinstance(raw, torch.Tensor):
                if hasattr(raw, 'share'): raw = raw.share
                elif hasattr(raw, '_tensor'): raw = raw._tensor
                else: break
            if isinstance(raw, torch.Tensor):
                return raw.to(target_device)
            return raw

        for i in range(2):
            # 1. 获取 Beaver Triples (每一轮都需要新的 Triples，因为 y 在变)
            # 注意：使用 cube_1 获取 x*y^2 模式
            l1, l2, l1_l2, l2_sq, l1_l2_sq = provider.cube_1(y_curr.size(), device=self.device, mode="xy_square")

            # 2. 强力清洗所有输入 (转为 GPU Raw Tensor)
            x_raw = to_raw_gpu(x_curr)
            y_raw = to_raw_gpu(y_curr) 
            l1_raw = to_raw_gpu(l1)
            l2_raw = to_raw_gpu(l2)
            l1_l2_raw = to_raw_gpu(l1_l2)
            l2_sq_raw = to_raw_gpu(l2_sq)
            l1_l2_sq_raw = to_raw_gpu(l1_l2_sq)

            # 3. Mask & Reveal
            masked_x_raw = x_raw + l1_raw
            masked_y_raw = y_raw + l2_raw
            
            stacked_raw = torch.stack([masked_x_raw, masked_y_raw], dim=0)
            
            import crypten.communicator as comm
            stacked_delta = comm.get().all_reduce(stacked_raw)
            
            delta1 = stacked_delta[0]
            delta2 = stacked_delta[1]

            # 4. Beaver 公式计算 z = x * y^2 (Scale = S^3)
            # 此时 res_raw 的 Scale 是 Sx * Sy * Sy
            res_raw = (
                delta1 * delta2 * delta2 
                - 2 * delta1 * delta2 * l2_raw 
                - delta2 * delta2 * l1_raw 
                + delta1 * l2_sq_raw 
                + 2 * delta2 * l1_l2_raw 
                - l1_l2_sq_raw
            )

            trunc_divisor = int(self.encoder.scale * y_curr.encoder.scale)
            
            # 执行整数除法进行截断
            res_raw = res_raw.div(trunc_divisor, rounding_mode='trunc')

            # 6. 封装为 MPCTensor (Scale = S)
            y3_term = y_curr.clone()
            
            # 注入数据
            target = y3_term._tensor
            if hasattr(target, 'share'):
                target.share = res_raw
            else:
                from crypten.mpc.primitives import ArithmeticSharedTensor
                new_ast = ArithmeticSharedTensor.from_shares(res_raw, precision=0)
                y3_term._tensor = new_ast
            
            # 设置 Scale (现在已经截断回标准 Scale 了)
            y3_term.encoder._scale = y_curr.encoder.scale

            # 7. 更新 y_curr: 2y - xy^2
            # 此时两个操作数 Scale 一致，直接相减
            y_curr = y_curr.mul(2.0) - y3_term
        l1, l2, l1_l2, l2_sq, l1_l2_sq = provider.cube_1(y_curr.size(), device=self.device, mode="xy_square")

        # 2. 强力清洗所有输入 (转为 GPU Raw Tensor)
        x_raw = to_raw_gpu(x_curr)
        y_raw = to_raw_gpu(y_curr) 
        l1_raw = to_raw_gpu(l1)
        l2_raw = to_raw_gpu(l2)
        l1_l2_raw = to_raw_gpu(l1_l2)
        l2_sq_raw = to_raw_gpu(l2_sq)
        l1_l2_sq_raw = to_raw_gpu(l1_l2_sq)
        mask_raw = to_raw_gpu(masked_share)  
        if mask_raw.dim() == x_raw.dim() - 1:
             mask_raw = mask_raw.unsqueeze(0)            
        # 3. Mask & Reveal
        masked_x_raw = x_raw + l1_raw
        masked_y_raw = y_raw + l2_raw
        
        stacked_raw = torch.cat([masked_x_raw, masked_y_raw, mask_raw], dim=0)
        stacked_delta = comm.get().all_reduce(stacked_raw)
        
        delta1 = stacked_delta[0]
        delta2 = stacked_delta[1]
        masked_sign_plain= stacked_delta[2]
        # 4. Beaver 公式计算 z = x * y^2 (Scale = S^3)
        # 此时 res_raw 的 Scale 是 Sx * Sy * Sy
        res_raw = (
            delta1 * delta2 * delta2 
            - 2 * delta1 * delta2 * l2_raw 
            - delta2 * delta2 * l1_raw 
            + delta1 * l2_sq_raw 
            + 2 * delta2 * l1_l2_raw 
            - l1_l2_sq_raw
        )

        trunc_divisor = int(self.encoder.scale * y_curr.encoder.scale)
        
        # 执行整数除法进行截断
        res_raw = res_raw.div(trunc_divisor, rounding_mode='trunc')

        # 6. 封装为 MPCTensor (Scale = S)
        y3_term = y_curr.clone()
        
        # 注入数据
        target = y3_term._tensor
        if hasattr(target, 'share'):
            target.share = res_raw
        else:
            from crypten.mpc.primitives import ArithmeticSharedTensor
            new_ast = ArithmeticSharedTensor.from_shares(res_raw, precision=0)
            y3_term._tensor = new_ast
        
        # 设置 Scale (现在已经截断回标准 Scale 了)
        y3_term.encoder._scale = y_curr.encoder.scale

        # 7. 更新 y_curr: 2y - xy^2
        # 此时两个操作数 Scale 一致，直接相减
        y_curr = y_curr.mul(2.0) - y3_term

        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)[0]

        y_curr= (y_curr[1]-y_curr[0]) * is_large- is_large

        # 循环结束，返回结果
        return y_curr
    elif method == "newer_time+":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        key_s = (use_k1, use_k2, 8.0)
        if key_s in _RECIPROCAL_PARAMS_TABLE:
            params = _RECIPROCAL_PARAMS_TABLE[key_s]
            a0 = params["a0"]
            poly_vals = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
        
        L_fit = 8.0
        full_poly = [0.0] + poly_vals

        poly_part = self.mul(full_poly[1])
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        y_final = _fourier_series_x2x3_time(
            self, 
            len(beta_sin_coeffs), 
            period, 
            alpha=a0,           
            beta_sin=beta_sin, 
            poly_coeffs=full_poly  
        )

        # === Step 5: 合成 (本地计算) ===
        y_curr = poly_part + y_final 
        x_curr=self
        target_device = self.device
        def to_raw_gpu(t):
            raw = t
            while not isinstance(raw, torch.Tensor):
                if hasattr(raw, 'share'): raw = raw.share
                elif hasattr(raw, '_tensor'): raw = raw._tensor
                else: break
            if isinstance(raw, torch.Tensor):
                return raw.to(target_device)
            return raw
        for i in range(3):
            # 1. 获取 Beaver Triples (每一轮都需要新的 Triples，因为 y 在变)
            # 注意：使用 cube_1 获取 x*y^2 模式
            provider = crypten.mpc.get_default_provider()
            l1, l2, l1_l2, l2_sq, l1_l2_sq = provider.cube_1(y_curr.size(), device=self.device, mode="xy_square")

            # 2. 强力清洗所有输入 (转为 GPU Raw Tensor)
            x_raw = to_raw_gpu(x_curr)
            y_raw = to_raw_gpu(y_curr) 
            l1_raw = to_raw_gpu(l1)
            l2_raw = to_raw_gpu(l2)
            l1_l2_raw = to_raw_gpu(l1_l2)
            l2_sq_raw = to_raw_gpu(l2_sq)
            l1_l2_sq_raw = to_raw_gpu(l1_l2_sq)

            # 3. Mask & Reveal
            masked_x_raw = x_raw + l1_raw
            masked_y_raw = y_raw + l2_raw
            
            stacked_raw = torch.stack([masked_x_raw, masked_y_raw], dim=0)
            
            import crypten.communicator as comm
            stacked_delta = comm.get().all_reduce(stacked_raw)
            
            delta1 = stacked_delta[0]
            delta2 = stacked_delta[1]

            # 4. Beaver 公式计算 z = x * y^2 (Scale = S^3)
            # 此时 res_raw 的 Scale 是 Sx * Sy * Sy
            res_raw = (
                delta1 * delta2 * delta2 
                - 2 * delta1 * delta2 * l2_raw 
                - delta2 * delta2 * l1_raw 
                + delta1 * l2_sq_raw 
                + 2 * delta2 * l1_l2_raw 
                - l1_l2_sq_raw
            )

            trunc_divisor = int(self.encoder.scale * y_curr.encoder.scale)
            
            # 执行整数除法进行截断
            res_raw = res_raw.div(trunc_divisor, rounding_mode='trunc')

            # 6. 封装为 MPCTensor (Scale = S)
            y3_term = y_curr.clone()
            
            # 注入数据
            target = y3_term._tensor
            if hasattr(target, 'share'):
                target.share = res_raw
            else:
                from crypten.mpc.primitives import ArithmeticSharedTensor
                new_ast = ArithmeticSharedTensor.from_shares(res_raw, precision=0)
                y3_term._tensor = new_ast
            
            # 设置 Scale (现在已经截断回标准 Scale 了)
            y3_term.encoder._scale = y_curr.encoder.scale

            # 7. 更新 y_curr: 2y - xy^2
            # 此时两个操作数 Scale 一致，直接相减
            y_curr = y_curr.mul(2.0) - y3_term

        return y_curr
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self, k1=1, k2=12, L=2048.0):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sqrt_method

    if method == "ideal":
        return crypten.cryptensor(torch.rsqrt(self.get_plain_text()), device=self.device)
    elif method == "NR":
        # Initialize using decent approximation
        if initial is None:
            y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
            y -= self.div(1024)
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iters):
            y = y.mul_(3 - self * y.square()).div_(2)
        return y
    elif method == "newer":
        use_k1 = k1 if k1 is not None else INVSQRT_K1
        use_k2 = k2 if k2 is not None else INVSQRT_K2
        
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L)
        
        period = 2 * L
        poly_part = self.polynomial(poly_body) + a0
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device)
        y0 = poly_part + _fourier_series_x3(self, len(beta_sin_coeffs), period, beta_sin=beta_sin)

        y0_cube = y0.cube()
        
        y1 = y0.mul(1.50131454).sub(self.mul(y0_cube.mul(0.500438180)))
        
        y2 = y1.mul(1.50000086).sub(self.mul(y1.cube().mul(0.499999)))
        
        return y2
    elif method == "newer_time+":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        key_s = (use_k1, use_k2, 8.0)
        if key_s in _INVSQRT_PARAMS_TABLE:
            params = _INVSQRT_PARAMS_TABLE[key_s]
            a0 = params["a0"]
            poly_vals = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
        L_fit=8.0
        full_poly = [0.0] + poly_vals
        # 3. 计算多项式部分
        poly_part =  self.mul(full_poly[1])
        # 4. 计算傅里叶部分
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        y_final = _fourier_series_x2x3_time(
            self, 
            len(beta_sin_coeffs), 
            period, 
            alpha=a0,          # 传入常数项
            beta_sin=beta_sin, 
            poly_coeffs=full_poly  # 【关键】传入多项式系数
        )
        # 5. 合成
        y0 =poly_part+y_final 
        
        # 9. Newton 迭代 
        # Iteration 1
        y0_cube = y0.cube()
        
        y1 = y0.mul(1.50131454).sub(self.mul(y0_cube.mul(0.500438180)))
        
        y2 = y1.mul(1.50000086).sub(self.mul(y1.cube().mul(0.499999)))
        return y2
    elif method == "newer_time":
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        key_s = (use_k1, use_k2, L_small)
        key_l = (use_k1, use_k2, L_large)
        # 2. 获取动态拟合参数
        # a0_s, poly_s, beta_s = _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_small, min_val=0.1)
        # a0_l, poly_l, beta_l = _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_large, min_val=L_small)
        if key_s in _INVSQRT_PARAMS_TABLE:
            params = _INVSQRT_PARAMS_TABLE[key_s]
            a0_s = params["a0"]
            poly_s = params["poly_body"]
            beta_s = params["beta_sin_coeffs"]
            
        else:
            a0_s, poly_s, beta_s= _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_small, min_val=0.1)

        if key_l in _INVSQRT_PARAMS_TABLE:
            params = _INVSQRT_PARAMS_TABLE[key_l]
            a0_l = params["a0"]
            poly_l = params["poly_body"]
            beta_l = params["beta_sin_coeffs"]
            
        else:
            a0_l, poly_l, beta_l= _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_large, min_val=L_small)
        period_s = 2 * L_small
        period_l = 2 * L_large
        if k1==3:
        # 3. 获取 Triples
            t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
            t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)
        else:
            t_s, u_s, v_s = provider.generate_hybrid_triple_1(self.size(), period_s, len(beta_s), device=device)
            t_l, u_l, v_l = provider.generate_hybrid_triple_1(self.size(), period_l, len(beta_l), device=device)
        # 4. 计算分段掩码
        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        u_cmp, v_cmp, w_cmp= provider.generate_additive_triple(self.size(),self.size(),"mul",device=device)
        eps_cmp_share = self - u_cmp
        delta_cmp_share =self+cmp_a - v_cmp-self     
        delta_s_share = (self + t_s + period_s)
        delta_l_share = (self + t_l + period_l)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        # 一次性通信 Reveal
        comm_block_1 = crypten.stack([
            eps_cmp_share, 
            delta_cmp_share,             
            delta_s_share,
            delta_l_share
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
        
        eps_cmp = plain_block_1[0]  
        delta_cmp = plain_block_1[1]  
        delta_s_plain_raw = plain_block_1[2]
        delta_l_plain_raw = plain_block_1[3]
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)    
        masked_share = (z_cmp + cmp_b)
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        # 6. 定义混合计算分支函数
        def _compute_hybrid_branch_x1(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
            delta_k = torch.stack([i * delta_mod for i in k_list])
            p = torch.sin(delta_k).to(device) # sin(kd)
            q = torch.cos(delta_k).to(device) # cos(kd)
            if isinstance(beta_list, list):
                # 确保转换到正确的设备 (device)
                beta_list = torch.tensor(beta_list, device=device, dtype=torch.float)
            beta_view = beta_list.view([-1] + [1] * self.dim())
            fourier_val = ((v * p - u * q) * beta_view).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            total = poly_val + fourier_val
            return total
        def _compute_hybrid_branch_x3(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
            delta_k = torch.stack([i * delta_mod for i in k_list])
            p = torch.sin(delta_k).to(device) # sin(kd)
            q = torch.cos(delta_k).to(device) # cos(kd)
            if isinstance(beta_list, list):
                # 确保转换到正确的设备 (device)
                beta_list = torch.tensor(beta_list, device=device, dtype=torch.float)
            beta_view = beta_list.view([-1] + [1] * self.dim())
            fourier_val = ((v * p - u * q) * beta_view).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            # 设置缩放因子防止溢出
            if i_flag == 0.0:
                s_sq = 32.0  
                s_cu = 32.0
            else:           
                s_sq = 256.0
                s_cu = 1024.0

            s_sq_pow2 = s_sq * s_sq
            s_cu_pow3 = s_cu * s_cu * s_cu 

            # --- 计算 x^2 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]
                
                # 变量缩放
                d_s2  = delta_raw / s_sq
                t_s2  = t / s_sq
                t2_s2 = t2 / s_sq_pow2
                d_s2_sq = d_s2.square()
                
                # x^2 = (d - t)^2 = d^2 - 2dt + t^2
                sq_term1 = d_s2_sq
                sq_term2 = t_s2 * (2 * d_s2) * -1.0
                sq_term3 = t2_s2
                x_sq_scaled = sq_term1 + sq_term2 + sq_term3
                
                # 融合乘法
                combined_c2 = c2 * s_sq_pow2
                poly_val += x_sq_scaled * combined_c2

            # --- 计算 x^3 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                # 变量缩放
                d_s3  = delta_raw / s_cu
                t_s3  = t / s_cu
                t2_s3 = t2 / (s_cu * s_cu)
                t3_s3 = t3 / (s_cu * s_cu * s_cu)
                d_s3_sq = d_s3.square()
                
                cu_term1 = d_s3 * d_s3_sq
                cu_term2 = t_s3 * (3 * d_s3_sq) * -1.0
                cu_term3 = t2_s3 * (3 * d_s3)
                x_cu_scaled = cu_term1 + cu_term2 + cu_term3 - t3_s3
                
                # 融合乘法
                combined_c3 = c3 * s_cu_pow3
                poly_val += x_cu_scaled * combined_c3

            total = poly_val + fourier_val
            return total
        if k1==1:
        # 7. 执行并行计算
            y_small = _compute_hybrid_branch_x1(
                delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
                t_s, u_s, v_s, "SMALL", 0.0
            )
            
            y_large = _compute_hybrid_branch_x1(
                delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
                t_l, u_l, v_l, "LARGE", 1.0
            )
        else:
        # 7. 执行并行计算
            y_small = _compute_hybrid_branch_x3(
                delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
                t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
            )
            
            y_large = _compute_hybrid_branch_x3(
                delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
                t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
            )
        y_curr = crypten.stack([y_small, y_large], dim=0)
        x_curr = crypten.stack([self, self], dim=0)
        # 8. 合并结果

        y0 = y_curr

        # 9. Newton 迭代 
        
        # Iteration 1
        y3_term1 = y0.cube() 
        half_x = x_curr.mul(0.500438180)
        
        # 计算被减数: 0.5 * x * y^3
        sub_term1 = half_x.mul(y3_term1)
        
        y1 = y0.mul(1.50131454) - sub_term1
        
        # Iteration 2
        y3_term2 = y1.cube().mul(0.999124984) 
        
        u_cmp, v_cmp, w_cmp= provider.generate_additive_triple(half_x.size(),y3_term2.size(),"mul",device=device)
        eps_cmp_share = half_x - u_cmp
        delta_cmp_share =y3_term2 - v_cmp
        mask=self+masked_share-self
        if mask.dim() == eps_cmp_share.dim() - 1:
            mask = mask.unsqueeze(0)
        comm_block_2 = crypten.cat([
            eps_cmp_share, 
            delta_cmp_share,             
            mask
        ], dim=0)
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
        eps_cmp = plain_block_2[0:2]  # 取前2行
        delta_cmp = plain_block_2[2:4]  # 取中间2行
        masked_sign_plain = plain_block_2[4]
       
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)    
        y2 = y1.mul(1.50000086) - z_cmp
        V = (masked_sign_plain > 0).float()
        is_large = cmp_r.mul(V).add(cmp_c)[0]
        is_small = 1.0 - is_large
        y_curr= (y2[1]-y2[0]) * is_large+is_small 
        y=self.mul(y_curr)
        return y_curr
    elif method == "newer_debug":
        
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 5  # 建议默认提高 K1
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        a0_s, poly_s, beta_s = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L=L_small, min_val=0.1)
        a0_l, poly_l, beta_l = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L=L_large, min_val=L_small)
        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
            x_real=self.get_plain_text()
            
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)
        is_small = 1.0 - is_large
        
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        def _compute_hybrid_branch(x,delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device) # sin(kd)
                q = torch.cos(delta_k).to(device) # cos(kd)
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            with crypten.no_grad():
                stacked_plain = stacked_shares.get_plain_text()
            # --- 计算 x^2 (使用 Fusion Mul) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]

                poly_val += x*x * c2

            # --- 计算 x^3 (使用 Fusion Mul) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                poly_val += x*x*x * c3 

            total = poly_val + fourier_val
            return total

        # 7. 执行并行计算
        y_small = _compute_hybrid_branch(
            x_real,delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        # Large Branch: 传入原始 t_l, 内部会进行除法缩放
        y_large = _compute_hybrid_branch(
            x_real,delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        # 8. 合并结果
        values = crypten.stack([y_small, y_large], dim=0)
        masks = crypten.stack([self+is_small-self, self+is_large-self], dim=0)
        products = values.mul(masks)
        y0 = products.sum(dim=0)
        _d("y0 (Initial)", y0)
        with crypten.no_grad():
            y0_real=y0.get_plain_text()
        # 9. Newton 迭代 

        y3_term1 = y0_real*y0_real*y0_real
        half_x = self.mul(0.500438180)
        
        sub_term1 = half_x.mul(y3_term1)
        
        y1 = y0.mul(1.50131454) - sub_term1
        with crypten.no_grad():
            y1_real=y1.get_plain_text()

        # Iteration 2 (逻辑保持不变，如果 y1 已经崩了，这里也会崩)
        y3_term2 = y1_real*y1_real*y1_real.mul(0.999124984) 
        y2 = y1.mul(1.50000086) - half_x.mul(y3_term2)
        return y2
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")


def sqrt(self, k1=1, k2=12):
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run
        sqrt_initial (tensor): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return self.inv_sqrt().mul(self)


def _eix(self):
    r"""Computes e^(i * self) where i is the imaginary unit.
    Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
    """
    iterations = cfg.functions.trig_iterations

    re = 1
    im = self.div(2**iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im.square()
    im *= 2

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul_(re)
        im._tensor *= 2
        re = a2 - b2

    return re, im


def cossin(self):
    r"""Computes cosine and sine of input via exp(i * x).

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return self._eix()


def cos(self):
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[0]


def sin(self):
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[1]


# Logistic Functions
def sigmoid(self, k1=1, k2=12,L=8.0):
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    If a valid method is given, this function will compute sigmoid
        using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with
        truncation and uses the identity:

    .. math::
        \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

    "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
        the reciprocal

    """  # noqa: W605
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.sigmoid(self.get_plain_text()), device=self.device)
    elif method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sign = 1 - 2 * ltz

        pos_input = self.mul(sign)
        denominator = pos_input.neg().exp().add(1)

        # TODO: Set these with configurable parameters
        with cfg.temp_override(
            {
                "functions.exp_iterations": 9,
                "functions.reciprocal_nr_iters": 3,
                "functions.reciprocal_all_pos": True,
                "functions.reciprocal_initial": 0.75,
            }
        ):
            pos_output = denominator.reciprocal()

        result = pos_output.where(1 - ltz, 1 - pos_output)
        # TODO: Support addition with different encoder scales
        # result = pos_output + ltz - 2 * pos_output * ltz
        return result
    elif method == "fourier":    
        m = cfg.functions.sigmoid_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.sigmoid_fs_terms

        # note that beta_cos = 0 for tanh
        alpha, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, alpha=alpha, beta_sin=beta_sin)
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        key = (use_k1, use_k2)

        if key in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
            print("这里拟合了")
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        if k1==1:
        # 3. 调用混合评估器 (x3 版本)
            mixed_part = _fourier_series_x_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        else :
            mixed_part = _fourier_series_x3_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        # 4. 合并结果
        final_res = linear_term + mixed_part
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        import crypten
        
        if isinstance(final_res, ArithmeticSharedTensor):
            # 创建空壳，注入数据，补全属性
            res = MPCTensor.__new__(MPCTensor)
            res._tensor = final_res
            res.ptype = crypten.mpc.arithmetic
            res.encoder = self.encoder
            return res
        return final_res
    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        key = (use_k1, use_k2)

        if key in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        if k1==1:
        # 3. 调用混合评估器 (x3 版本)
            mixed_part = _fourier_series_x(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        else :
            mixed_part = _fourier_series_x3(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        # 4. 合并结果
        final_res = linear_term + mixed_part

        return final_res
    elif method == "newer_time+":
        import crypten.communicator as comm
        import math
        z_share = self  
        threshold = 12.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        sigmoid_k1, sigmoid_k2, sigmoid_L = k1, k2, 12.0 
        # a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", sigmoid_k1, sigmoid_k2, sigmoid_L)
        key = (k1, k2)

        if key in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        full_poly = [0.0] + poly_body
        period = 2 * sigmoid_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        (
            (a_all, b_all, r_all, c_all),           # CMP Aux
            (u_cmp, v_cmp, w_cmp),                  # Triple 1
            (t_sigmoid, u_sigmoid, v_sigmoid, t2_sigmoid, t3_sigmoid),  # Hybrid
            (u_silu, v_silu, w_silu),               # Triple 2
            (u_final, v_final, w_final)             # Triple 3
        ) = provider.generate_gelu_offline_batch(
            self.size(), double_size, period, len(beta_sin_coeffs), device=device
        )
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share =a_all - v_cmp     
        delta_sigmoid_share = (z_share + t_sigmoid + period)._tensor
        # ================== 🛠️ 调试代码结束 ==================
        comm_block_1 = type(delta_sigmoid_share).cat([
            eps_cmp_share,                 # (2, Size)
            delta_cmp_share,               # (2, Size)
            delta_sigmoid_share.unsqueeze(0) # (1, Size)
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
        # 解包 Round 1
        eps_cmp = plain_block_1[0:2]     # CMP 的 epsilon
        delta_cmp = plain_block_1[2:4]   # CMP 的 delta
        delta_sigmoid_plain_1 = plain_block_1[4] # Sigmoid 的 delta
        
        delta_real = delta_sigmoid_plain_1.float()
        delta_mod = delta_real
        delta_raw = delta_mod - period
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_sigmoid * p - u_sigmoid * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
        
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            d_sq = delta_raw.square()
            cube_t1 = delta_raw * d_sq
            cube_t2 = t_sigmoid * (3 * d_sq)
            cube_t3 = t2_sigmoid * (3 * delta_raw)
            z_cube = cube_t1 + cube_t2 + cube_t3 + t3_sigmoid
            
            z_cube = z_cube.div(self.encoder.scale ** 2) # 建议加上这步
            poly_res += z_cube * c3
            
        sigmoid_out = poly_res + fourier_res
        # (B) 计算 CMP 的乘法结果 z = inputs * a_all
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)
        # --- 4. 第二轮计算与通信 (Round 2) ---
        masked_share = (z_cmp + b_all)

        # 打包发送
        comm_block_2 = type(delta_sigmoid_share).cat([
            masked_share,                   # (2, Size)
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
   
        masked_plain = plain_block_2[0:2] 
        
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0] # x > 12
        is_neg_large = indicators[1] # x < -12
        
        silu_product=sigmoid_out
        # 3. 现在的 silu_product 就是一个标准的 MPCTensor 了
        middle_mask = 1.0 - is_pos_large - is_neg_large
        
        eps_final_1 = silu_product._tensor - u_final[0]
        delta_final_1 = middle_mask - v_final[0] 
        eps_final_2 = z_share._tensor - u_final[1]
        delta_final_2 = is_pos_large - v_final[1]

        comm_block_3 = type(eps_cmp_share).cat([
            eps_final_1,
            delta_final_1,
            eps_final_2,
            delta_final_2
        ], dim=0)
        with crypten.no_grad():
            plain_block_3 = comm_block_3.get_plain_text()
            
        # --- 解包 ---
        eps_f1 = plain_block_3[0]
        delta_f1 = plain_block_3[1]
        eps_f2 = plain_block_3[2]
        delta_f2 = plain_block_3[3]
        
        # --- Beaver 重构 (含精度修正) ---
        
        # 计算 Term 1
        term4_f1 = (eps_f1.double() * delta_f1.double()).float()
        term_1 = w_final[0] + v_final[0].mul(eps_f1) + u_final[0].mul(delta_f1) + term4_f1
        
        # 计算 Term 3
        term4_f2 = (eps_f2.double() * delta_f2.double()).float()
        term_3 = w_final[1] + v_final[1].mul(eps_f2) + u_final[1].mul(delta_f2) + term4_f2
        
        # --- 最终加和 ---
        final_result = term_1 + term_3
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        import crypten
        
        if isinstance(final_result, ArithmeticSharedTensor):
            # 创建空壳，注入数据，补全属性
            res = MPCTensor.__new__(MPCTensor)
            res._tensor = final_result
            res.ptype = crypten.mpc.arithmetic
            res.encoder = self.encoder
            return res
        return final_result
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")

def tanh(self, k1=1, k2=12, L=7.0):
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    If a valid method is given, this function will compute tanh using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with truncation.

    .. math::
        tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

    where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
    The approximation is truncated to +/-1 outside [-1, 1].

    Args:
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    """
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.tanh(self.get_plain_text()), device=self.device)
    if method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh_terms
        coeffs = crypten.common.util.chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return out.hardtanh()
    elif method == "poly":
        drelu_x = self >= 0
        sign_x = 2 * drelu_x - 1
        abs_x = sign_x * self
        do_poly = abs_x < 2.95
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([1.1950192,-0.49313435,0.0737858,-0.00147019]) - 0.01758266
        out = sign_x * (do_poly * (poly_x - 1) + 1)
        return out
    elif method == "fourier":
        m = cfg.functions.tanh_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.tanh_fs_terms

        # note that alpha, beta_cos = 0 for tanh
        _, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, beta_sin=beta_sin)
    elif method == "ode":
        iter_num = cfg.functions.tanh_ode_iter_num
        x = self / iter_num
        y = self.new(torch.zeros_like(self.data), device=self.device)
        for _ in range(iter_num):
            y += (1 - y * y) * x
        return y
    elif method == "newer_time":
        
        use_k1 = k1 if k1 is not None else TANH_K1
        use_k2 = k2 if k2 is not None else TANH_K2

        key = (use_k1, use_k2)

        if key in _TANH_PARAMS_TABLE:
            params = _TANH_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", use_k1, use_k2, L)
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        if k1==1:
        # 3. 调用混合评估器 (x3 版本)
            mixed_part = _fourier_series_x_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        else :
            mixed_part = _fourier_series_x3_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        # 4. 合并结果
        final_res = linear_term + mixed_part

        return final_res

    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else TANH_K1
        use_k2 = k2 if k2 is not None else TANH_K2

        key = (use_k1, use_k2)

        if key in _TANH_PARAMS_TABLE:
            params = _TANH_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", use_k1, use_k2, L)
        linear_term = self.mul(poly_body[0])

        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        if k1==1:
            mixed_part = _fourier_series_x(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        else :
            mixed_part = _fourier_series_x3(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        # 4. 合并结果
        final_res = linear_term + mixed_part

        return final_res
    elif method == "newer_time+":
        import crypten.communicator as comm
        import math
        z_share = self  
        threshold = 12.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        sigmoid_k1, sigmoid_k2, sigmoid_L = k1, k2, 12.0 
        # a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", sigmoid_k1, sigmoid_k2, sigmoid_L)
        key = (k1, k2)

        if key in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        full_poly = [0.0] + poly_body
        period = 2 * sigmoid_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        (
            (a_all, b_all, r_all, c_all),           # CMP Aux
            (u_cmp, v_cmp, w_cmp),                  # Triple 1
            (t_sigmoid, u_sigmoid, v_sigmoid, t2_sigmoid, t3_sigmoid),  # Hybrid
            (u_silu, v_silu, w_silu),               # Triple 2
            (u_final, v_final, w_final)             # Triple 3
        ) = provider.generate_gelu_offline_batch(
            self.size(), double_size, period, len(beta_sin_coeffs), device=device
        )
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share =a_all - v_cmp     
        delta_sigmoid_share = (z_share + t_sigmoid + period)._tensor
        # ================== 🛠️ 调试代码结束 ==================
        comm_block_1 = type(delta_sigmoid_share).cat([
            eps_cmp_share,                 # (2, Size)
            delta_cmp_share,               # (2, Size)
            delta_sigmoid_share.unsqueeze(0) # (1, Size)
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
        # 解包 Round 1
        eps_cmp = plain_block_1[0:2]     # CMP 的 epsilon
        delta_cmp = plain_block_1[2:4]   # CMP 的 delta
        delta_sigmoid_plain_1 = plain_block_1[4] # Sigmoid 的 delta
        
        delta_real = delta_sigmoid_plain_1.float()
        delta_mod = delta_real
        delta_raw = delta_mod - period
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_sigmoid * p - u_sigmoid * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
        
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            d_sq = delta_raw.square()
            cube_t1 = delta_raw * d_sq
            cube_t2 = t_sigmoid * (3 * d_sq)
            cube_t3 = t2_sigmoid * (3 * delta_raw)
            z_cube = cube_t1 + cube_t2 + cube_t3 + t3_sigmoid
            
            z_cube = z_cube.div(self.encoder.scale ** 2) # 建议加上这步
            poly_res += z_cube * c3
            
        sigmoid_out = poly_res + fourier_res
        # (B) 计算 CMP 的乘法结果 z = inputs * a_all
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)
        # --- 4. 第二轮计算与通信 (Round 2) ---
        masked_share = (z_cmp + b_all)

        # 打包发送
        comm_block_2 = type(delta_sigmoid_share).cat([
            masked_share,                   # (2, Size)
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
   
        masked_plain = plain_block_2[0:2] 
        
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0] # x > 12
        is_neg_large = indicators[1] # x < -12
        
        silu_product=sigmoid_out
        # 3. 现在的 silu_product 就是一个标准的 MPCTensor 了
        middle_mask = 1.0 - is_pos_large - is_neg_large
        
        eps_final_1 = silu_product._tensor - u_final[0]
        delta_final_1 = middle_mask - v_final[0] 
        eps_final_2 = z_share._tensor - u_final[1]
        delta_final_2 = is_pos_large - v_final[1]

        comm_block_3 = type(eps_cmp_share).cat([
            eps_final_1,
            delta_final_1,
            eps_final_2,
            delta_final_2
        ], dim=0)
        with crypten.no_grad():
            plain_block_3 = comm_block_3.get_plain_text()
            
        # --- 解包 ---
        eps_f1 = plain_block_3[0]
        delta_f1 = plain_block_3[1]
        eps_f2 = plain_block_3[2]
        delta_f2 = plain_block_3[3]
        
        # --- Beaver 重构 (含精度修正) ---
        
        # 计算 Term 1
        term4_f1 = (eps_f1.double() * delta_f1.double()).float()
        term_1 = w_final[0] + v_final[0].mul(eps_f1) + u_final[0].mul(delta_f1) + term4_f1
        
        # 计算 Term 3
        term4_f2 = (eps_f2.double() * delta_f2.double()).float()
        term_3 = w_final[1] + v_final[1].mul(eps_f2) + u_final[1].mul(delta_f2) + term4_f2
        
        # --- 最终加和 ---
        final_result = term_1 + term_3
        return final_result
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def _chebyshev_polynomials(self, terms):
    r"""Evaluates odd degree Chebyshev polynomials at x

    Chebyshev Polynomials of the first kind are defined as

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        self (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return crypten.stack(polynomials)

def _fourier_series_x3_time(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
        r"""
        Hybrid Evaluator (x3 Only - Production Version):
        - Communication: Reveals (delta + period) to handle scale and positive range.
        - Fourier: Uses (delta + period) directly (periodicity holds).
        - Polynomial: Recovers (delta) locally to ensure x^3 correctness.
        """

        if beta_cos is not None:
            raise NotImplementedError("Fourier series with cosine is currently not supported")
        if beta_sin is None:
            raise ValueError("beta_sin cannot be None")
        
        device = self.device
        beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        
        # 1. 获取混合三元组
        provider = crypten.mpc.get_default_provider()
        
        # 这里可能会产生通信 (取决于 provider 实现)
        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)

        with crypten.no_grad():
            delta_share = self + t + period 
            
            # 这里一定会产生通信 (Reveal 操作)
            delta_mod = delta_share.get_plain_text() 
            
        delta_k = torch.stack([i * delta_mod for i in k])
        p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

        fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

        poly_res = 0
        if poly_coeffs is not None:
            delta_raw = delta_mod - period

            if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
                c3 = poly_coeffs[3]

                delta_sq = delta_raw.square()
                
                cube_term1 = delta_raw * delta_sq
                
                # 这里涉及密文计算，可能会有通信
                cube_term2 = t * (3 * delta_sq)
                cube_term3 = t2 * (3 * delta_raw)
                
                x_cube = cube_term1 + cube_term2 + cube_term3 + t3
                
                poly_res = poly_res + (x_cube * c3)

        if isinstance(poly_res, (int, float)) and poly_res == 0:
            final_res = fourier_res
        else:
            final_res = fourier_res + poly_res

        if alpha is not None:
            final_res = final_res + alpha
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        if isinstance(final_res, ArithmeticSharedTensor):
            result = MPCTensor.__new__(MPCTensor)
            result._tensor = final_res
            return result
        return final_res

def _fourier_series_x_time(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
        r"""
        Hybrid Evaluator (x3 Only - Production Version):
        - Communication: Reveals (delta + period) to handle scale and positive range.
        - Fourier: Uses (delta + period) directly (periodicity holds).
        - Polynomial: Recovers (delta) locally to ensure x^3 correctness.
        """

        if beta_cos is not None:
            raise NotImplementedError("Fourier series with cosine is currently not supported")
        if beta_sin is None:
            raise ValueError("beta_sin cannot be None")
        
        device = self.device
        beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

        provider = crypten.mpc.get_default_provider()
        
        t, u, v = provider.generate_hybrid_triple_1(self.size(), period, terms, device=device)

        with crypten.no_grad():
            delta_share = self + t + period 
            # 这里一定会产生通信 (Reveal 操作)
            delta_mod = delta_share.get_plain_text() 
            
        delta_k = torch.stack([i * delta_mod for i in k])
        p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

        fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)
        
        
        poly_res = 0
        if isinstance(poly_res, (int, float)) and poly_res == 0:
            final_res = fourier_res
        else:
            final_res = fourier_res + poly_res
        if alpha is not None:
            final_res = final_res + alpha
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        if isinstance(final_res, ArithmeticSharedTensor):
            result = MPCTensor.__new__(MPCTensor)
            result._tensor = final_res
            return result
        return final_res

def _fourier_series_x(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
        r"""
        Hybrid Evaluator (x3 Only - Production Version):
        - Communication: Reveals (delta + period) to handle scale and positive range.
        - Fourier: Uses (delta + period) directly (periodicity holds).
        - Polynomial: Recovers (delta) locally to ensure x^3 correctness.
        """

        if beta_cos is not None:
            raise NotImplementedError("Fourier series with cosine is currently not supported")
        if beta_sin is None:
            raise ValueError("beta_sin cannot be None")
        
        device = self.device
        beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

        provider = crypten.mpc.get_default_provider()
        
        t, u, v = provider.generate_hybrid_triple_1(self.size(), period, terms, device=device)

        with crypten.no_grad():
            delta_share = self + t + period 
            # 这里一定会产生通信 (Reveal 操作)
            delta_mod = delta_share.get_plain_text() 
            
        delta_k = torch.stack([i * delta_mod for i in k])
        p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

        fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)
        
        
        poly_res = 0
        if isinstance(poly_res, (int, float)) and poly_res == 0:
            final_res = fourier_res
        else:
            final_res = fourier_res + poly_res
        if alpha is not None:
            final_res = final_res + alpha
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        if isinstance(final_res, ArithmeticSharedTensor):
            result = MPCTensor.__new__(MPCTensor)
            result._tensor = final_res
            return result
        return final_res

def _fourier_series_x3(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
        r"""
        Hybrid Evaluator (x3 Only - Production Version):
        - Communication: Reveals (delta + period) to handle scale and positive range.
        - Fourier: Uses (delta + period) directly (periodicity holds).
        - Polynomial: Recovers (delta) locally to ensure x^3 correctness.
        """
        if beta_cos is not None:
            raise NotImplementedError("Fourier series with cosine is currently not supported")
        if beta_sin is None:
            raise ValueError("beta_sin cannot be None")
        
        device = self.device
        beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
        k = [i * 2 * math.pi / period for i in range(1, terms + 1)]
        
        provider = crypten.mpc.get_default_provider()
        
        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)
        
        with crypten.no_grad():
            # 掩码揭示 delta = x + t + period
            delta_share = self + t + period 
            delta_mod = delta_share.get_plain_text() 
            
        delta_k = torch.stack([i * delta_mod for i in k])
        p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

        term1 = v * p
        term1 = term1.div(self.encoder.scale) # 定点数乘法 Scale 修正

        term2 = u * q
        term2 = term2.div(self.encoder.scale) # 定点数乘法 Scale 修正

        combo = term1 - term2
        
        fourier_res = (combo * beta_sin).sum(dim=0)

        poly_res = 0
        if poly_coeffs is not None:
            delta_raw = delta_mod - period

            if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
                c3 = poly_coeffs[3]

                delta_sq = delta_raw.square()
                
                cube_term1 = delta_raw * delta_sq
                
                cube_term2 = t * (-3 * delta_sq)
                
                cube_term3 = t2 * (3 * delta_raw)

                cube_term4 = t3.neg()
                
                # 合并
                x_cube = cube_term1 + cube_term2 + cube_term3 + cube_term4
                
                poly_res = poly_res + (x_cube * c3)

        if isinstance(poly_res, (int, float)) and poly_res == 0:
            final_res = fourier_res
        else:
            final_res = fourier_res + poly_res

        if alpha is not None:
            final_res = final_res + alpha
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        if isinstance(final_res, ArithmeticSharedTensor):
            result = MPCTensor.__new__(MPCTensor)
            result._tensor = final_res
            return result
        return final_res
def _fourier_series_x2x3_time(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (Final Correct Version)
    """
    # 参数校验
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    provider = crypten.mpc.get_default_provider()
    
    # 1. 获取预计算的三元组/辅助参数 (加密状态)
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)
    
    # 2. 计算 Delta (揭示掩码后的差值)
    with crypten.no_grad():
        delta_share = self + t +period
        delta_mod = delta_share.get_plain_text() 
    delta_raw = delta_mod - period


    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    poly_res = 0
    if poly_coeffs is not None:
        delta_sq = delta_raw.square() 
        # 计算 x^2
        if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
            c2 = poly_coeffs[2]
            
            sq_term1 = delta_sq                    # d^2
            sq_term2 = t * (2 * delta_raw)         # 2dt (Crypto * Plain)
            
            sq_term3 = t2                          # t^2
            
            x_square = sq_term1 - sq_term2 + sq_term3
            poly_res = poly_res + (x_square * c2)

        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]
            
            cube_term1 = delta_raw * delta_sq      # d^3
            cube_term2 = t * (3 * delta_sq)        # 3d^2t
            cube_term3 = t2 * (3 * delta_raw)      # 3dt^2

            x_cube = cube_term1 - cube_term2 + cube_term3 - t3
            poly_res = poly_res + (x_cube * c3)

    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
    from crypten.mpc.mpc import MPCTensor
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
    if isinstance(final_res, ArithmeticSharedTensor):
        result = MPCTensor.__new__(MPCTensor)
        result._tensor = final_res
        return result
    return final_res

def _fourier_series_x2x3(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (Final Correct Version)
    """
    # 参数校验
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    provider = crypten.mpc.get_default_provider()
    
    # 1. 获取预计算的三元组/辅助参数 (加密状态)
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)
    
    # 2. 计算 Delta (揭示掩码后的差值)
    with crypten.no_grad():
        delta_share = self + t +period
        delta_mod = delta_share.get_plain_text() 
        x_plain=self.get_plain_text() 
    delta_raw = delta_mod - period


    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    poly_res = 0
    if poly_coeffs is not None:
        # 计算 x^2
        if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
            c2 = poly_coeffs[2]
            poly_res = poly_res + x_plain*x_plain*c2

        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]
            
            poly_res = poly_res + x_plain*x_plain*x_plain*c3

    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
    from crypten.mpc.mpc import MPCTensor
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
    if isinstance(final_res, ArithmeticSharedTensor):
        result = MPCTensor.__new__(MPCTensor)
        result._tensor = final_res
        return result
    return final_res
def erf(self,tensor=None,k1=1, k2=12,L=5.0):
    r"""
    Approximates the error function of the input tensor.
    """
    method = cfg.functions.erf_method

    if method == "ideal":
        return crypten.cryptensor(torch.erf(tensor.get_plain_text()), device=tensor.device)
    elif method == "taylor":
        iters = cfg.functions.erf_iterations

        output = tensor.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(tensor.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
        # NOTE: This approximation is not unstable for large tensor values.
    elif method == "tanh":
        return tanh(math.sqrt(4 / math.pi) * (tensor + 0.044715 * tensor.pow(3)))
    elif method == "fourier":
        period = cfg.functions.erf_fs_period
        width = period / 2
        terms = cfg.functions.erf_fs_terms

        # note that alpha, beta_cos = 0 for erf
        _, _, beta_sin = crypten.common.util.fourier_series(torch.erf, width, terms)
        return _fourier_series(tensor, terms, period, beta_sin=beta_sin)
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else ERF_K1
        use_k2 = k2 if k2 is not None else ERF_K2

        # a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("erf", use_k1, use_k2, L)
        key = (use_k1, use_k2)
        # 1. 现场计算拟合系数
        if key in _ERF_PARAMS_TABLE:
            params = _ERF_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs= _get_dynamic_params_odd("erf", use_k1, use_k2, L)
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        if k1==1:
        # 3. 调用混合评估器 (x3 版本)
            mixed_part = _fourier_series_x_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        else :
            mixed_part = _fourier_series_x3_time(
                self, 
                terms=len(beta_sin_coeffs), 
                period=period, 
                alpha=a0, 
                beta_sin=beta_sin, 
                poly_coeffs=full_poly
            )
        # 4. 合并结果
        final_res = linear_term + mixed_part

        return final_res

    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else ERF_K1
        use_k2 = k2 if k2 is not None else ERF_K2

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("erf", use_k1, use_k2, L)
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        # 3. 调用混合评估器 (x3 版本)
        mixed_part = _fourier_series_x3(
            self, 
            terms=len(beta_sin_coeffs), 
            period=period, 
            alpha=a0, 
            beta_sin=beta_sin, 
            poly_coeffs=full_poly
        )
        
        # 4. 合并结果
        final_res = linear_term + mixed_part
        return final_res
    else:
        raise ValueError(f"Unrecognized method {method} for erf")

def _diff_gelu(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="none") - torch.nn.functional.relu(x))

def _diff_gelu_tanh(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="tanh") - torch.nn.functional.relu(x))

def _diff_silu(x):
    return torch.sign(x) * (torch.nn.functional.silu(x) - torch.nn.functional.relu(x))

def gelu(self, approximate="none", k1=1, k2=12):
    r"""Compute the Gaussian error linear unit of a tensor"""
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.gelu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.gelu(self.get_plain_text(), approximate=approximate), device=self.device)
    elif method == "fourier":
        period = cfg.functions.gelu_fs_period
        width = period / 2
        terms = cfg.functions.gelu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        if approximate == "tanh":
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu_tanh, width, terms)
            beta_sin = torch.tensor([-0.0817,-0.0812,-0.0424,-0.0175,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        else:
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu, width, terms)
            beta_sin = torch.tensor([-0.0818,-0.0809,-0.0424,-0.0176,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    elif method == "secformer":
        # set erf_fs_period: 20, erf_fs_terms: 7
        b0, b1 = self > -1.7 * math.sqrt(2), self < 1.7 * math.sqrt(2)
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    elif method == "poly":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 3
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([-0.55386347,0.5658561,-0.19719836,0.02328962]) + 0.00410626
        return relu_x + do_poly * poly_x
    elif method == "bolt":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 2.7
        # Motzkin's polynomial preprocessing
        g = [0.14439048359960427, -0.7077117131613893, 4.5702822654246535, -8.15444702051307, 16.382265425072532]
        poly_x = (g[0] * abs_x + g[1]) * abs_x + g[2]
        poly_x = (poly_x + g[0] * abs_x + g[3]) * abs_x + g[4] + 0.5 * self
        # The g's provided by BOLT are wrong, uncomment the following line to get the correct approximation
        #poly_x = abs_x.polynomial([-0.53798164612714154,0.5410550166368381,-0.18352506127082727,0.020848611754127593]) + 0.001620808531841547
        return relu_x + do_poly * poly_x
    elif method == "erf":
        # set erf_fs_period: 16, erf_fs_terms: 5
        b0, b1 = self > -2, self < 2
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    elif method == "newer":
        use_k1 = k1 if k1 is not None else GELU_K1
        use_k2 = k2 if k2 is not None else GELU_K2

        threshold = 4.0 
        stacked_inputs = crypten.stack([self - threshold, self.neg() - threshold])
        indicators = stacked_inputs.od_sign() 
        
        is_pos_large = indicators[0]
        is_neg_large = indicators[1] 
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0

        # --- 2. 计算 Tanh 参数 ---
        sqrt_2_pi = math.sqrt(2 / math.pi)
        
        x3 = self.cube()
        
        # 公式: sqrt(2/pi) * (0.044715 * x^3 + x)
        inner_poly = x3.mul(0.044715).add(self).mul(sqrt_2_pi)
        
        tanh_out = inner_poly.tanh(k1=use_k1, k2=use_k2)
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))

        # --- 3. 最终合成 ---
        return self.mul(is_pos_large) + mid_res.mul(is_mid)
    elif method == "newer_debug":
        import crypten.communicator as comm
        import math
        # 记录初始状态
        last_stats = crypten.get_communication_stats()

        def log_step(step_name):
            nonlocal last_stats
            # 获取当前总状态
            current = crypten.get_communication_stats()
            # 计算增量 (Delta)
            diff_bytes = current["bytes"] - last_stats["bytes"]
            diff_rounds = current["rounds"] - last_stats["rounds"]
            
            # 仅 Rank 0 打印，避免刷屏
            if comm.get().get_rank() == 0:
                print(f"📡 [Step: {step_name:<20}] "
                      f"Bytes: {diff_bytes / 1024 / 1024:.4f} MB | "
                      f"Rounds: {diff_rounds}")
            
            # 更新状态锚点
            last_stats = current
        z_share = self  
        inv_sqrt_2 = 0.7071067811865475
        z=z_share*inv_sqrt_2
        threshold = 5.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        erf_k1, erf_k2, erf_L = k1, k2, 5.0 
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("erf", erf_k1, erf_k2, erf_L)
        full_poly = [0.0] + poly_body
        period = 2 * erf_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        u_cmp, v_cmp, w_cmp = provider.generate_additive_triple(double_size, double_size,"mul",device=device)
        t_erf, u_erf, v_erf, t2_erf, t3_erf = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        u_gelu, v_gelu, w_gelu = provider.generate_additive_triple(self.size(),self.size(),"mul", device=device)
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share =a_all - v_cmp     
        delta_erf_share = (z + t_erf + period)._tensor
        comm_block_1 = type(delta_erf_share).cat([
            eps_cmp_share,                 # (2, Size)
            delta_cmp_share,               # (2, Size)
            delta_erf_share.unsqueeze(0) # (1, Size)
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            x_plain=z.get_plain_text()   
        # 解包 Round 1
        eps_cmp = plain_block_1[0:2]     # CMP 的 epsilon
        delta_cmp = plain_block_1[2:4]   # CMP 的 delta
        delta_erf_plain_1 = plain_block_1[4] # erf 的 delta
        
        delta_real = delta_erf_plain_1.float()
        delta_mod = delta_real
        delta_raw = delta_mod - period
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_erf * p - u_erf * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        if len(full_poly) > 1:
            poly_res += z * full_poly[1]
        
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            poly_res += x_plain*x_plain*x_plain * c3
            
        erf_out = (poly_res + fourier_res+1.0)*0.5
        # (B) 计算 CMP 的乘法结果 z = inputs * a_all
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)
        # --- 4. 第二轮计算与通信 (Round 2) ---
        masked_share = (z_cmp + b_all)

        eps_gelu_share = self._tensor - u_gelu
        delta_gelu_share = erf_out._tensor - v_gelu
        # 打包发送
        comm_block_2 = type(delta_erf_share).cat([
            masked_share,                   # (2, Size)
            eps_gelu_share.unsqueeze(0),    # (1, Size)
            delta_gelu_share.unsqueeze(0)   # (1, Size)
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
            
        masked_plain = plain_block_2[0:2] 
        eps_gelu = plain_block_2[2]       
        delta_gelu = plain_block_2[3]     
        
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0] # x > 12
        is_neg_large = indicators[1] # x < -12
        
        gelu_product_ast = (w_gelu + v_gelu.mul(eps_gelu) + u_gelu.mul(delta_gelu) + eps_gelu * delta_gelu)
        gelu_product=(self+gelu_product_ast-self)
        # 3. 现在的 gelu_product 就是一个标准的 MPCTensor 了
        middle_mask = 1.0 - is_pos_large - is_neg_large
        lhs_batch = crypten.stack([self+gelu_product-self,z_share], dim=0)
        rhs_batch = crypten.stack([self+middle_mask-self,self+is_pos_large-self], dim=0)
        products_batch = lhs_batch.mul(rhs_batch) 
   
        term_1 = products_batch[0]
        term_3 = products_batch[1]
        
        final_gelu = term_1 + term_3
        
        return final_gelu
    elif method == "newer_time":
        import math
        import crypten.communicator as comm
        
        # 记录初始状态
        last_stats = crypten.get_communication_stats()

        # def log_step(step_name):
        #     nonlocal last_stats
        #     # 获取当前总状态
        #     current = crypten.get_communication_stats()
        #     # 计算增量 (Delta)
        #     diff_bytes = current["bytes"] - last_stats["bytes"]
        #     diff_rounds = current["rounds"] - last_stats["rounds"]
            
        #     # 仅 Rank 0 打印，避免刷屏
        #     if comm.get().get_rank() == 0:
        #         print(f"📡 [Step: {step_name:<20}] "
        #               f"Bytes: {diff_bytes / 1024 / 1024:.4f} MB | "
        #               f"Rounds: {diff_rounds}")
            
        #     # 更新状态锚点
        #     last_stats = current
        z_share = self
        inv_sqrt_2 = 0.7071067811865475
        threshold = 5.0
        scale = self.encoder.scale 
        
        erf_k1, erf_k2, erf_L = k1, k2, 5.0 
        key = (k1, k2)

        if key in _ERF_PARAMS_TABLE:
            params = _ERF_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("erf", erf_k1, erf_k2, erf_L)
        full_poly = [0.0] + poly_body
        period = 2 * erf_L
        
        device = self.device
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        
        input_size = self.size()
        double_size = (2,) + input_size

        provider = crypten.mpc.get_default_provider()

        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)

        u_cmp, v_cmp, w_cmp = provider.generate_additive_triple(
            double_size, double_size, "mul", device=device
        )
        terms=len(beta_sin_coeffs)
        if k1==1:
            t_erf, u_erf, v_erf = provider.generate_hybrid_triple_1(
                self.size(), period, terms, device=device
            )
        else:
            t_erf, u_erf, v_erf, t2_erf, t3_erf = provider.generate_hybrid_triple(
                self.size(), period, terms, device=device
            )
        u_gelu, v_gelu, w_gelu = provider.generate_additive_triple(
            self.size(), self.size(), "mul", device=device
        )

        u_final, v_final, w_final = provider.generate_additive_triple(
            double_size, double_size, "mul", device=device
        )
        z = z_share * inv_sqrt_2
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share = a_all - v_cmp
        delta_erf_share = (z + t_erf + period)._tensor
        
        comm_block_1 = type(eps_cmp_share).cat([
            eps_cmp_share,                 
            delta_cmp_share,               
            delta_erf_share.unsqueeze(0) 
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            
        eps_cmp = plain_block_1[0:2]     
        delta_cmp = plain_block_1[2:4]   
        delta_erf_plain_1 = plain_block_1[4] 
        
        delta_real = delta_erf_plain_1.float() / scale 
        delta_mod = delta_real
        delta_raw = delta_mod - period
        
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        
        fourier_res = ((v_erf * p - u_erf * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        if k1==1:
            if len(full_poly) > 1:
                poly_res += z * full_poly[1]

        else:
            if len(full_poly) > 1:
                poly_res += z * full_poly[1]
            
            if len(full_poly) > 3 and full_poly[3] != 0:
                c3 = full_poly[3]
                d_sq = delta_raw.square()
                cube_t1 = delta_raw * d_sq
                cube_t2 = t_erf * (3 * d_sq)
                cube_t3 = t2_erf * (3 * delta_raw)
                z_cube = cube_t1 + cube_t2 + cube_t3 + t3_erf
                poly_res += z_cube * c3

            
        erf_out = (poly_res + fourier_res + 1.0) * 0.5
        term4_cmp = (eps_cmp.double() * delta_cmp.double()).float()
        z_cmp = w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + term4_cmp

        masked_share = z_cmp + b_all
        
        eps_gelu_share = z_share._tensor - u_gelu
        delta_gelu_share = erf_out._tensor - v_gelu
        
        comm_block_2 = type(eps_cmp_share).cat([
            masked_share,                   
            eps_gelu_share.unsqueeze(0),    
            delta_gelu_share.unsqueeze(0)   
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
            
        masked_plain = plain_block_2[0:2] 
        eps_gelu = plain_block_2[2]       
        delta_gelu = plain_block_2[3]     
        
        V = (masked_plain > 0).float()
        is_pos_large = V[0] 
        is_neg_large = V[1] 
        middle_mask = 1.0 - is_pos_large - is_neg_large
        
        term4_gelu = (eps_gelu.double() * delta_gelu.double()).float()
        gelu_product_ast = w_gelu + v_gelu.mul(eps_gelu) + u_gelu.mul(delta_gelu) + term4_gelu
        
        eps_final_1 = gelu_product_ast - u_final[0]
        delta_final_1 = middle_mask - v_final[0] 
        
        eps_final_2 = z_share._tensor - u_final[1]
        delta_final_2 = is_pos_large - v_final[1] 

        comm_block_3 = type(eps_cmp_share).cat([
            eps_final_1,
            delta_final_1,
            eps_final_2,
            delta_final_2
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_3 = comm_block_3.get_plain_text()
            
        eps_f1 = plain_block_3[0]
        delta_f1 = plain_block_3[1]
        eps_f2 = plain_block_3[2]
        delta_f2 = plain_block_3[3]
        
        # Calc Term 1
        term4_f1 = (eps_f1.double() * delta_f1.double()).float()
        term_1 = w_final[0] + v_final[0].mul(eps_f1) + u_final[0].mul(delta_f1) + term4_f1
        
        # Calc Term 3
        term4_f2 = (eps_f2.double() * delta_f2.double()).float()
        term_3 = w_final[1] + v_final[1].mul(eps_f2) + u_final[1].mul(delta_f2) + term4_f2
        
        # 最终加和
        final_gelu = term_1 + term_3
        from crypten.mpc.mpc import MPCTensor
        from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
        if isinstance(final_gelu, ArithmeticSharedTensor):
            res = MPCTensor.__new__(MPCTensor)
            res._tensor = final_gelu
            res.ptype = crypten.mpc.arithmetic
            res.encoder = self.encoder
            return res
        return final_gelu
    elif method == "newer_debug_1":
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]
        
        t, t2, t3 = provider.cube(self.size(), device=self.device, mode="cube")
        
        masked_pos = (self - threshold).mul(a1).add(b1)
        masked_neg = (self.neg() - threshold).mul(a2).add(b2)
        delta_share = self - t
        
        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            x_plain = self.get_plain_text()
        # --- 4. 解包与重构 ---
        V_pos = (all_plain[0] > 0).float()
        V_neg = (all_plain[1] > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        is_neg_large = r2.mul(V_neg).add(c2)
        
        delta_plain = all_plain[2] 
        x3 = x_plain*x_plain*x_plain

        # --- 5. Tanh 计算 ---
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0
        inner_poly_ast = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        tanh_out = inner_poly_ast.tanh(k1=k1, k2=k2)
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))

        # --- 6. 最终合成 ---
        effective_share_ast = self.mul(is_pos_large)+mid_res * is_mid 
        gelu_res = effective_share_ast
        return gelu_res
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")

def silu(self, k1=1, k2=12,L=12.0):
    r"""Compute the Sigmoid linear unit of a tensor with global variable support"""
    # 自动判定 method
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.silu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.silu(self.get_plain_text()), device=self.device)
    elif method == "fourier":
        period = cfg.functions.silu_fs_period
        width = period / 2
        terms = cfg.functions.silu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        #_, _, beta_sin = crypten.common.util.fourier_series(_diff_silu, width, terms)
        beta_sin = torch.tensor([-0.1299, -0.1220, -0.0743, -0.0394, -0.0216, -0.0118, \
                                 -0.0074, -0.0044, -0.0033, -0.0021, -0.0018, -0.0011], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    
    elif method == "newer":
        use_k1 = k1 if k1 is not None else SILU_K1
        use_k2 = k2 if k2 is not None else SILU_K2

        # --- Debug 准备：预先获取输入明文 ---
        x_plain = self.get_plain_text()

        # --- 1. 分段逻辑 ---
        threshold = 12.0
        stacked_inputs = crypten.stack([self - threshold, self.neg() - threshold])
        # 必须对 stacked_inputs 调用 od_sign 以获得两个指示器
        indicators = stacked_inputs.od_sign()
        
        is_pos_large = indicators[0]
        is_neg_large = indicators[1] 
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0
        fit_L=threshold+1.0
        # --- 2. 现场拟合 Sigmoid 部分 ---
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("sigmoid", use_k1, use_k2, fit_L)
        
        period = 2 * fit_L
        poly_part = self.polynomial(poly_body) + a0
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        fourier_part = _fourier_series_x3(self, len(beta_sin_coeffs), period, beta_sin=beta_sin)
        
        sig_x_approx = poly_part + fourier_part
        # 计算中间段：x * sigmoid(x)
        mid_res = self.mul(sig_x_approx)

        # --- 3. 最终结果合成 ---
        # x > 5 -> x; x < -5 -> 0; mid -> x * sig_approx
        res = self.mul(is_pos_large) + mid_res.mul(is_mid)

        # # ==================== TOP 10 ERRORS DEBUG START ====================
        # # 计算标准真值 (使用 torch.nn.functional)
        import torch.nn.functional as F_debug # 使用别名避免冲突
        true_silu = F_debug.silu(x_plain)
        
        # 提取中间项明文用于分析
        res_plain = res.get_plain_text()
        is_pos_plain = is_pos_large.get_plain_text()
        is_mid_plain = is_mid.get_plain_text()
        sig_x_plain = sig_x_approx.get_plain_text()
        mid_res_plain = mid_res_plain = mid_res.get_plain_text()


        return res
    elif method == "newer_debug":
        z_share = self  
        threshold = 12.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        import math
        sigmoid_k1, sigmoid_k2, sigmoid_L = k1, k2, 12.0 
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", sigmoid_k1, sigmoid_k2, sigmoid_L)
        full_poly = [0.0] + poly_body
        period = 2 * sigmoid_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()

        # def inspect_var(name, var):
        #     # 1. 获取类型
        #     v_type = type(var)
            
        #     # 2. 获取明文 (处理 MPCTensor 和 ArithmeticSharedTensor 的差异)
        #     if hasattr(var, 'get_plain_text'):
        #         # MPCTensor 或 ArithmeticSharedTensor
        #         with crypten.no_grad():
        #             v_plain = var.get_plain_text()
        #     elif hasattr(var, 'share') and hasattr(var.share, 'get_plain_text'):
        #         # 某些封装情况
        #         with crypten.no_grad():
        #             v_plain = var.share.get_plain_text()
        #     else:
        #         # 普通 Tensor
        #         v_plain = var
            
        #     # 获取当前 rank
        #     import crypten.communicator as comm
        #     current_rank = comm.get().get_rank()
        #     if current_rank != 0:
        #         return
        #     # 打印当前 rank 的信息（所有 rank 都打印，便于调试）
        #     print(f"[Rank {current_rank}] 👉 [{name}]")
        #     print(f"[Rank {current_rank}]     Type:      {v_type}")
            
        #     # 安全地展平并预览数据
        #     try:
        #         if isinstance(v_plain, torch.Tensor):
        #             flat_data = v_plain.detach().cpu().view(-1).tolist()
        #         else:
        #             flat_data = [float(v_plain)]  # 标量
        #         preview = flat_data[:5]
        #         total = len(flat_data)
        #         print(f"[Rank {current_rank}]     Plain[:5]: {preview} ... (Total {total})")
        #     except Exception as e:
        #         print(f"[Rank {current_rank}]     Plain: <failed to extract: {e}>")
            
        #     print(f"[Rank {current_rank}] " + "-" * 20)

        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        u_cmp, v_cmp, w_cmp = provider.generate_additive_triple(double_size, double_size,"mul",device=device)
        t_sigmoid, u_sigmoid, v_sigmoid, t2_sigmoid, t3_sigmoid = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        u_silu, v_silu, w_silu = provider.generate_additive_triple(self.size(),self.size(),"mul", device=device)
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share =a_all - v_cmp     
        delta_sigmoid_share = (z_share + t_sigmoid + period)._tensor
        # ================== 🛠️ 调试代码结束 ==================
        comm_block_1 = type(delta_sigmoid_share).cat([
            eps_cmp_share,                 # (2, Size)
            delta_cmp_share,               # (2, Size)
            delta_sigmoid_share.unsqueeze(0) # (1, Size)
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            x_plain=self.get_plain_text()
            
        # 解包 Round 1
        eps_cmp = plain_block_1[0:2]     # CMP 的 epsilon
        delta_cmp = plain_block_1[2:4]   # CMP 的 delta
        delta_sigmoid_plain_1 = plain_block_1[4] # Sigmoid 的 delta
        scale_2_20 = 1048576.0 
        
        delta_real = delta_sigmoid_plain_1.float()
        delta_mod = delta_real
        delta_raw = delta_mod - period
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_sigmoid * p - u_sigmoid * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
        
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            poly_res += x_plain*x_plain*x_plain * c3
            
        sigmoid_out = poly_res + fourier_res
        # (B) 计算 CMP 的乘法结果 z = inputs * a_all
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)
        # --- 4. 第二轮计算与通信 (Round 2) ---
        masked_share = (z_cmp + b_all)

        eps_silu_share = self._tensor - u_silu
        delta_silu_share = sigmoid_out._tensor - v_silu
        # 打包发送
        comm_block_2 = type(delta_sigmoid_share).cat([
            masked_share,                   # (2, Size)
            eps_silu_share.unsqueeze(0),    # (1, Size)
            delta_silu_share.unsqueeze(0)   # (1, Size)
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
            
        masked_plain = plain_block_2[0:2] 
        eps_silu = plain_block_2[2]       
        delta_silu = plain_block_2[3]     
        
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0] # x > 12
        is_neg_large = indicators[1] # x < -12
        
        silu_product_ast = (w_silu + v_silu.mul(eps_silu) + u_silu.mul(delta_silu) + eps_silu * delta_silu)
        silu_product=(self+silu_product_ast-self)
        # 3. 现在的 silu_product 就是一个标准的 MPCTensor 了
        middle_mask = 1.0 - is_pos_large - is_neg_large
        lhs_batch = crypten.stack([self+silu_product-self,z_share], dim=0)
        rhs_batch = crypten.stack([self+middle_mask-self,self+is_pos_large-self], dim=0)
        products_batch = lhs_batch.mul(rhs_batch) 
   
        term_1 = products_batch[0]
        term_3 = products_batch[1]
        
        final_silu = term_1 + term_3
        
        return final_silu
    elif method == "newer_time":
        import crypten.communicator as comm
        import math
        # 记录初始状态
        # last_stats = crypten.get_communication_stats()

        # def log_step(step_name):
        #     nonlocal last_stats
        #     # 获取当前总状态
        #     current = crypten.get_communication_stats()
        #     # 计算增量 (Delta)
        #     diff_bytes = current["bytes"] - last_stats["bytes"]
        #     diff_rounds = current["rounds"] - last_stats["rounds"]
            
        #     # 仅 Rank 0 打印，避免刷屏
        #     if comm.get().get_rank() == 0:
        #         print(f"📡 [Step: {step_name:<20}] "
        #               f"Bytes: {diff_bytes / 1024 / 1024:.4f} MB | "
        #               f"Rounds: {diff_rounds}")
            
        #     # 更新状态锚点
        #     last_stats = current

        z_share = self  
        threshold = 12.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        sigmoid_k1, sigmoid_k2, sigmoid_L = k1, k2, 12.0 
        # a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", sigmoid_k1, sigmoid_k2, sigmoid_L)
        key = (k1, k2)

        if key in _SIGMOID_PARAMS_TABLE:
            params = _SIGMOID_PARAMS_TABLE[key]
            a0 = params["a0"]
            poly_body = params["poly_body"]
            beta_sin_coeffs = params["beta_sin_coeffs"]
            
        else:
            a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        full_poly = [0.0] + poly_body
        period = 2 * sigmoid_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        terms=len(beta_sin_coeffs)
        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)

        u_cmp, v_cmp, w_cmp = provider.generate_additive_triple(
            double_size, double_size, "mul", device=device
        )
        if k1==1:
            t_sigmoid, u_sigmoid, v_sigmoid = provider.generate_hybrid_triple_1(
                self.size(), period, terms, device=device
            )
        else:
            t_sigmoid, u_sigmoid, v_sigmoid, t2_sigmoid, t3_sigmoid = provider.generate_hybrid_triple(
                self.size(), period, terms, device=device
            )
        u_silu, v_silu, w_silu = provider.generate_additive_triple(
            self.size(), self.size(), "mul", device=device
        )

        u_final, v_final, w_final = provider.generate_additive_triple(
            double_size, double_size, "mul", device=device
        )
        inputs_cmp = crypten.stack([z_share - threshold, z_share.neg() - threshold])
        eps_cmp_share = inputs_cmp._tensor - u_cmp
        delta_cmp_share =a_all - v_cmp     
        delta_sigmoid_share = (z_share + t_sigmoid + period)._tensor
        # ================== 🛠️ 调试代码结束 ==================
        comm_block_1 = type(delta_sigmoid_share).cat([
            eps_cmp_share,                 # (2, Size)
            delta_cmp_share,               # (2, Size)
            delta_sigmoid_share.unsqueeze(0) # (1, Size)
        ], dim=0)

        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
        # 解包 Round 1
        eps_cmp = plain_block_1[0:2]     # CMP 的 epsilon
        delta_cmp = plain_block_1[2:4]   # CMP 的 delta
        delta_sigmoid_plain_1 = plain_block_1[4] # Sigmoid 的 delta
        
        delta_real = delta_sigmoid_plain_1.float()
        delta_mod = delta_real
        delta_raw = delta_mod - period
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_sigmoid * p - u_sigmoid * q) * beta_view).sum(dim=0)
        poly_res = a0
        if k1==1:
            if len(full_poly) > 1:
                poly_res += z_share * full_poly[1]

        else:
            if len(full_poly) > 1:
                poly_res += z_share * full_poly[1]
                        
            if len(full_poly) > 3 and full_poly[3] != 0:
                c3 = full_poly[3]
                d_sq = delta_raw.square()
                cube_t1 = delta_raw * d_sq
                cube_t2 = t_sigmoid * (3 * d_sq)
                cube_t3 = t2_sigmoid * (3 * delta_raw)
                z_cube = cube_t1 + cube_t2 + cube_t3 + t3_sigmoid
                
                z_cube = z_cube.div(self.encoder.scale ** 2) # 建议加上这步
                poly_res += z_cube * c3
        sigmoid_out = poly_res + fourier_res
        # (B) 计算 CMP 的乘法结果 z = inputs * a_all
        z_cmp = (w_cmp + v_cmp.mul(eps_cmp) + u_cmp.mul(delta_cmp) + eps_cmp * delta_cmp)
        # --- 4. 第二轮计算与通信 (Round 2) ---
        masked_share = (z_cmp + b_all)

        eps_silu_share = self._tensor - u_silu
        delta_silu_share = sigmoid_out._tensor - v_silu
        # 打包发送
        comm_block_2 = type(delta_sigmoid_share).cat([
            masked_share,                   # (2, Size)
            eps_silu_share.unsqueeze(0),    # (1, Size)
            delta_silu_share.unsqueeze(0)   # (1, Size)
        ], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
   
        masked_plain = plain_block_2[0:2] 
        eps_silu = plain_block_2[2]       
        delta_silu = plain_block_2[3]     
        
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0] # x > 12
        is_neg_large = indicators[1] # x < -12
        
        silu_product_ast = (w_silu + v_silu.mul(eps_silu) + u_silu.mul(delta_silu) + eps_silu * delta_silu)
        silu_product=(self+silu_product_ast-self)
        # 3. 现在的 silu_product 就是一个标准的 MPCTensor 了
        middle_mask = 1.0 - is_pos_large - is_neg_large
        
        eps_final_1 = silu_product._tensor - u_final[0]
        delta_final_1 = middle_mask - v_final[0] 
        eps_final_2 = z_share._tensor - u_final[1]
        delta_final_2 = is_pos_large - v_final[1]

        comm_block_3 = type(eps_cmp_share).cat([
            eps_final_1,
            delta_final_1,
            eps_final_2,
            delta_final_2
        ], dim=0)
        with crypten.no_grad():
            plain_block_3 = comm_block_3.get_plain_text()
            
        # --- 解包 ---
        eps_f1 = plain_block_3[0]
        delta_f1 = plain_block_3[1]
        eps_f2 = plain_block_3[2]
        delta_f2 = plain_block_3[3]
        
        # --- Beaver 重构 (含精度修正) ---
        
        # 计算 Term 1
        term4_f1 = (eps_f1.double() * delta_f1.double()).float()
        term_1 = w_final[0] + v_final[0].mul(eps_f1) + u_final[0].mul(delta_f1) + term4_f1
        
        # 计算 Term 3
        term4_f2 = (eps_f2.double() * delta_f2.double()).float()
        term_3 = w_final[1] + v_final[1].mul(eps_f2) + u_final[1].mul(delta_f2) + term4_f2
        
        # --- 最终加和 ---
        final_result = term_1 + term_3
        return final_result
    
    elif method == "newer_time_1":
        # Debug 工具
        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 12.0 
        L_fit = max(float(L), threshold + 1.0)
        period = 2 * L_fit
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("sigmoid", use_k1, use_k2, L=L_fit)
        full_poly_coeffs = [0.0] + poly_body 
        
        # --- [优化修改开始] 批量生成随机数 (Batch Generation) ---
        # 1. 构造双倍尺寸：(2, Batch_Size, ...)
        double_size = (2,) + self.size()
        
        # 2. 一次性生成两组辅助参数
        # 返回的 a_all, b_all 等都是 shape 为 (2, ...) 的 Share
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        # 3. 切片分离 (Slicing)
        # 注意：Crypten 的切片操作通常是零拷贝的，非常快
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]
        # --- [优化修改结束] ---

        diff_pos = self - threshold
        masked_pos = diff_pos.mul(a1).add(b1)

        diff_neg = self + threshold
        masked_neg = diff_neg.mul(a2).add(b2) # M2 = (x+12)*a2 + b2

        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, len(beta_sin_coeffs), device=device)
        delta_share = self + t + period

        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            
        masked_pos_plain = all_plain[0]
        masked_neg_plain = all_plain[1]
        delta_plain_raw = all_plain[2]

        V_pos = (masked_pos_plain > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        
        V_neg = (masked_neg_plain > 0).float()
        is_larger_than_neg12 = r2.mul(V_neg).add(c2)
        
        is_neg_large = 1.0 - is_larger_than_neg12
        
        is_mid = 1.0 - is_pos_large - is_neg_large

        # --- B. 还原拟合结果 (Hybrid Fitting) ---
        delta_exact = delta_plain_raw
        delta_mod = torch.fmod(delta_plain_raw, period)
        delta_mod[delta_mod < 0] += period
        
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        
        # --- 计算 Sigmoid 近似值 (内联 Hybrid Branch) ---
        fourier_val = 0
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([i * delta_mod for i in k_list])
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_tensor = beta_sin.view([-1] + [1] * self.dim())
        fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
        
        delta_raw = delta_exact - period
        poly_val = a0
        
        # 线性项
        if len(full_poly_coeffs) > 1:
            poly_val += self * full_poly_coeffs[1]
            
        delta_sq = delta_raw.square()
        
        # x^2 = (d-t)^2
        if len(full_poly_coeffs) > 2 and full_poly_coeffs[2] != 0:
            c2 = full_poly_coeffs[2]
            term_sq = delta_sq - t * (2 * delta_raw) + t2
            poly_val += term_sq * c2
            
        # x^3 = (d-t)^3
        if len(full_poly_coeffs) > 3 and full_poly_coeffs[3] != 0:
            c3 = full_poly_coeffs[3]
            cube_term1 = delta_raw * delta_sq
            cube_term2 = t * (3 * delta_sq)
            cube_term3 = t2 * (3 * delta_raw)
            term_cu = cube_term1 - cube_term2 + cube_term3 - t3
            poly_val += term_cu * c3
            
        sig_approx = poly_val + fourier_val

        effective_share = sig_approx.mul(is_mid)+ is_pos_large

        silu_res = self.mul(effective_share)
        return silu_res
    elif method == "newer_debug_1":
        # Debug 工具
        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 12.0 
        L_fit = max(float(L), threshold + 1.0)
        period = 2 * L_fit
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L=L_fit)
        full_poly_coeffs = [0.0] + poly_body 
        
        double_size = (2,) + self.size()
        
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]

        diff_pos = self - threshold
        masked_pos = diff_pos.mul(a1).add(b1)

        diff_neg = self + threshold
        masked_neg = diff_neg.mul(a2).add(b2) 

        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, len(beta_sin_coeffs), device=device)
        delta_share = self + t + period

        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            
        masked_pos_plain = all_plain[0]
        masked_neg_plain = all_plain[1]
        delta_plain_raw = all_plain[2]

        V_pos = (masked_pos_plain > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        
        V_neg = (masked_neg_plain > 0).float()
        is_larger_than_neg12 = r2.mul(V_neg).add(c2)
        
        is_neg_large = 1.0 - is_larger_than_neg12
        
        is_mid = 1.0 - is_pos_large - is_neg_large
        delta_exact = delta_plain_raw
        delta_mod = torch.fmod(delta_plain_raw, period)
        delta_mod[delta_mod < 0] += period
        
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        
        fourier_val = 0
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([i * delta_mod for i in k_list])
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_tensor = beta_sin.view([-1] + [1] * self.dim())
        fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
        
        delta_raw = delta_exact - period
        poly_val = a0
        
        # 线性项
        if len(full_poly_coeffs) > 1:
            poly_val += self * full_poly_coeffs[1]
            
        delta_sq = delta_raw.square()
        with crypten.no_grad():
            x_plain = self.get_plain_text()
        # x^2 = (d-t)^2
        if len(full_poly_coeffs) > 2 and full_poly_coeffs[2] != 0:
            c2 = full_poly_coeffs[2]
            poly_val += x_plain*x_plain * c2
            
        # x^3 = (d-t)^3
        if len(full_poly_coeffs) > 3 and full_poly_coeffs[3] != 0:
            c3 = full_poly_coeffs[3]
            poly_val += x_plain*x_plain *x_plain * c3
            
        sig_approx = poly_val + fourier_val

        effective_share = sig_approx.mul(is_mid)+ is_pos_large

        silu_res = self.mul(effective_share)
        return silu_res
    else:
        raise ValueError(f"Unrecognized method {method} for silu")

def softmax(self, dim, k1=3, k2_exp=12, k2_recip=12, **kwargs):
    r"""Compute the softmax of a tensor's elements along a given dimension"""
    if kwargs.get('k2') is not None:
        if k2_exp is None: k2_exp = kwargs['k2']
        if k2_recip is None: k2_recip = kwargs['k2']
        
    if k1 is not None or k2_exp is not None:
        method = "newer"
    else:
        method = cfg.functions.softmax_method
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.ones_like((self.data)))

    if self.size(dim) == 1:
        return self.new(torch.ones_like(self.data))

    if method == "ideal":
        return crypten.cryptensor(torch.softmax(self.get_plain_text(), dim=dim), device=self.device)
    if method == "reciprocal":
        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    elif method == "ode":
        iter_num = cfg.functions.softmax_ode_iter_num
        clip = cfg.functions.softmax_ode_clip
        upper, lower = cfg.functions.softmax_ode_ub, cfg.functions.softmax_ode_lb

        if clip:
            # clip the input within the range [lower, upper] for numerical stability
            diff = crypten.cat([self - upper, lower - self]).relu().split(self.shape[0])#.split([1,1])
            self += diff[1] - diff[0]

        # initialize ode approximation
        x = self / iter_num
        g = self.new(torch.ones_like(self.data) / self.size(dim), device=self.device)

        # compute ode update formula
        for _ in range(iter_num):
            g += (x - g.mul(x).sum(dim=dim).unsqueeze(-1)).squeeze(-1) * g
        return g
    elif method == "newer":
    
        # --- Step 1 & 2: OD-Sign ---
        alpha_star = -4.0 
        beta_star = 10.0
        x_input = self
        
        # 这里的 stack 是本地操作
        cmp_inputs = crypten.stack([alpha_star - x_input, x_input - beta_star], dim=0)
        
        # od_sign 是核心通信点
        indicators = od_sign(cmp_inputs) 

        # --- Step 3: Indicators split & Mult ---
        t1, t2 = indicators[0], indicators[1]
        diff_lower = alpha_star - x_input
        diff_upper = x_input - beta_star
        
        # 这里的 stack 和加减法通常是本地的
        mult_inputs_a = crypten.stack([self+t1-self, self+t2-self], dim=0)
        mult_inputs_b = crypten.stack([diff_lower, diff_upper], dim=0)
        
        # mul 是核心通信点 (1轮)
        mult_results = mult_inputs_a.mul(mult_inputs_b)

        # --- Step 4: Local Clamping ---
        # 这一步全是加减法，理论上通信量应为 0
        term_lower, term_upper = mult_results[0], mult_results[1]
        x_clamped = x_input + term_lower - term_upper - 12

        # --- Step 5: Exp ---
        # exp 是核心通信点 (多轮)
        t3 = x_clamped.exp(k1=k1, k2=k2_exp, fit_min=alpha_star-beta_star-2, fit_max=-2)

        # --- Step 6 & 7: Sum & Reciprocal ---
        # sum(dim) 是本地归约操作 (Rounds=0)
        t4 = t3.sum(dim, keepdim=True)
        
        # reciprocal 是核心通信点 (多轮)
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            t5 = t4.reciprocal(k1=k1, k2=k2_recip)

        # --- Step 8: Final Multiply ---
        # mul 是核心通信点 (1轮)
        y = t3.mul(t5)

        return y
    else:
        raise ValueError(f"Unrecognized method {method} for softmax")

def log_softmax(self, dim, **kwargs):
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.zeros((), device=self.device))

    if self.size(dim) == 1:
        return self.new(torch.zeros_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - normalize_term.log()
    return result

def od_sign(self):
    """
    Implements Protocol 4: CMP+ (Optimized Comparison) -> Sign Bit (0 or 1)
    """
    provider = crypten.mpc.get_default_provider()
    # 1. 获取辅助数据 (ArithmeticSharedTensor)
    a_share, b_share, r_share, c_share = provider.generate_cmp_aux(self.size(), device=self.device)
    
    # 2. 计算 Masked = x * a + b
    # 直接利用 MPCTensor 对底层 Share 的运算支持
    masked = self.mul(a_share).add(b_share)
    
    # 3. Reveal 掩码 (使用 get_plain_text 自动处理 scale)
    with crypten.no_grad():
        masked_plain = masked.get_plain_text()
    
    # 4. 明文比较 (得到 0.0 或 1.0)
    V = (masked_plain > 0).float()
    
    # 5. 计算 Indicator = r * V + c
    # r_share 是 AST，V 是 Tensor，结果自动为 AST
    indicator = r_share.mul(V).add(c_share)
    
    return indicator


def odrelu(self):
    """
    Implements Protocol 4: CMP+ (Optimized Comparison) -> ReLU
    """
    provider = crypten.mpc.get_default_provider()
    a_share, b_share, r_share, c_share = provider.generate_cmp_aux(self.size(), device=self.device)
    
    masked = self.mul(a_share).add(b_share)
    
    with crypten.no_grad():
        masked_plain = masked.get_plain_text()
    
    V = (masked_plain > 0).float()
    
    indicator = r_share.mul(V).add(c_share)
    
    return self.mul(indicator)


def odrelu(self):
    """
    Debug Version: Protocol 4 CMP+ -> ReLU
    复用上面的逻辑，但在最后多一步 x * indicator
    """
    indicator = self.od_sign()
    
    res = self.mul(indicator)
        
    return res
def _get_dynamic_params(func_type, K1, K2, L, min_val=None, max_val=None): 
    # 1. 基础配置
    num_samples = 20000 
    L_val = float(L)
    ODD_FUNCS = ["tanh,sigmoid"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # [FIX] 强制使用 float32 进行采样
    x_fit = np.linspace(start, end, num_samples, dtype=np.float32)

    # 3. 目标函数 Ground Truth
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    # [FIX] 强制 y_fit 为 float32
    y_fit = target_map[f_key](x_fit).astype(np.float32)

    # 4. 构建设计矩阵 X (强制 float32)
    X_list = [np.ones_like(x_fit, dtype=np.float32)] 
    for k in range(1, K1 + 1):
        X_list.append(x_fit.astype(np.float32) ** k)
    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float32)
        X_list.append(np.sin(arg)) 
    X = np.vstack(X_list).T.astype(np.float32)

    if func_type == "exp":
        safe_y = y_fit + 1e-20 
        weights = 1.0 / (safe_y ** 2)
        W = np.sqrt(weights)[:, np.newaxis].astype(np.float32)
        
        X_weighted = X * W
        y_weighted = y_fit * W.flatten()
        
        coeffs, residuals, rank, s = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    else:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y_fit, rcond=None)

    # [FIX] 结果强制转为 float32 (虽然 list 会存 float，但在转 tensor 时对齐)
    coeffs = coeffs.astype(np.float32)

    # 6. 结果提取
    a0 = coeffs[0].item()
    poly_coeffs_raw = coeffs[1 : 1 + K1]
    beta_sin = coeffs[1 + K1 :].tolist()
    
    poly_body = []
    for i, c in enumerate(poly_coeffs_raw):
        power = i + 1
        if is_odd_mode and (power % 2 == 0):
            poly_body.append(0.0)
        else:
            poly_body.append(c.item())
            
    return a0, poly_body, beta_sin

def _get_dynamic_params_odd(func_type, K1, K2, L, min_val=None, max_val=None): 
    # 1. 基础配置
    num_samples = 20000 
    L_val = float(L)
    ODD_FUNCS = ["tanh", "sigmoid", "erf"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # [FIX] 强制 float32
    x_fit = np.linspace(start, end, num_samples, dtype=np.float32)

    # 3. 目标函数
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "erf": lambda x: scipy.special.erf(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    # [FIX] 强制 float32
    y_fit = target_map[f_key](x_fit).astype(np.float32)

    # 4. 构建设计矩阵
    X_list = [np.ones_like(x_fit, dtype=np.float32)]
    
    if is_odd_mode:
        active_powers = [p for p in range(1, K1 + 1) if p % 2 != 0]
    else:
        active_powers = list(range(1, K1 + 1))

    for k in active_powers:
        X_list.append(x_fit.astype(np.float32) ** k)

    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float32)
        X_list.append(np.sin(arg)) 
        
    X = np.vstack(X_list).T.astype(np.float32)

    if func_type == "exp":
        weights = 1.0 / (y_fit + 1e-12)
        W = np.sqrt(weights)[:, np.newaxis].astype(np.float32)
        X_weighted = X * W
        y_weighted = y_fit * W.flatten()
        coeffs, residuals, rank, s = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    else:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y_fit, rcond=None)

    # [FIX] 结果强制转为 float32
    coeffs = coeffs.astype(np.float32)

    a0 = coeffs[0].item()
    
    num_poly_fitted = len(active_powers)
    
    poly_coeffs_computed = coeffs[1 : 1 + num_poly_fitted]
    
    beta_sin = coeffs[1 + num_poly_fitted :].tolist()

    poly_body = []
    
    coeff_iter = iter(poly_coeffs_computed)
    
    for k in range(1, K1 + 1):
        if k in active_powers:
            poly_body.append(next(coeff_iter).item())
        else:
            poly_body.append(0.0)
            
    return a0, poly_body, beta_sin

def _get_dynamic_params_1(func_type, K1, K2, L, min_val=None, max_val=None): 
    import numpy as np
    # 1. 基础配置
    num_samples = 20000 
    L_val = float(L)
    
    ODD_FUNCS = ["tanh", "sigmoid"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # --- 混合采样策略 ---
    if func_type in ["inv_sqrt", "inv", "reciprocal"] and start > 1e-9:
        if L_val < 20.0:
            ratio_geo = 0.7
        else:
            ratio_geo = 0.8
        n_geo = int(num_samples * ratio_geo)
        n_lin = num_samples - n_geo
        
        # [FIX] 强制 float32
        x_geo = np.geomspace(start, end, n_geo, dtype=np.float32)
        x_lin = np.linspace(start, end, n_lin, dtype=np.float32)
        x_fit = np.concatenate([x_geo, x_lin])
        x_fit = np.sort(x_fit)
    else:
        # [FIX] 强制 float32
        x_fit = np.linspace(start, end, num_samples, dtype=np.float32)

    # 3. 目标函数
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    # [FIX] 强制 float32
    y_fit = target_map[f_key](x_fit).astype(np.float32)

    # 4. 构建设计矩阵
    X_list = [np.ones_like(x_fit, dtype=np.float32)] 
    
    for k in range(1, K1 + 1):
        X_list.append(x_fit.astype(np.float32) ** k)
    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float32)
        X_list.append(np.sin(arg)) 
    X = np.vstack(X_list).T.astype(np.float32)

    # 加权逻辑
    if L_val < 20.0:
        alpha = 0.1 
    else:
        alpha = 1e-6
    
    weights = None

    if func_type == "exp":
        weights = 1.0 / (np.abs(y_fit) + 1e-15)
        
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        weights = 1.0 / (y_fit ** 2 + 1e-15)

    if weights is not None:
        W_sqrt = np.sqrt(weights)[:, np.newaxis].astype(np.float32)
        X_w = X * W_sqrt
        y_w = y_fit * np.sqrt(weights).astype(np.float32)
    else:
        X_w = X
        y_w = y_fit

    # 5. 求解
    n_features = X.shape[1]
    
    XT_w = X_w.T
    XTX_w = XT_w @ X_w
    A = XTX_w + alpha * np.eye(n_features, dtype=np.float32)
    b = XT_w @ y_w
    
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
    
    # [FIX] 结果强制转为 float32
    coeffs = coeffs.astype(np.float32)

    # 6. 结果提取
    a0 = coeffs[0].item()
    poly_coeffs_raw = coeffs[1 : 1 + K1]
    beta_sin = coeffs[1 + K1 :].tolist()
    
    poly_body = []
    for i, c in enumerate(poly_coeffs_raw):
        power = i + 1
        if is_odd_mode and (power % 2 == 0):
            poly_body.append(0.0)
        else:
            poly_body.append(c.item())
            
    return a0, poly_body, beta_sin

class IgnoreEncodings:
    """Context Manager to ignore tensor encodings"""
    def __init__(self, list_of_tensors):
        self.list_of_tensors = list_of_tensors
        # 记录原始的 scale
        self.encodings_cache = [tensor.encoder.scale for tensor in list_of_tensors]

    def __enter__(self):
        # 暂时将 scale 设为 1，进行纯整数运算
        for tensor in self.list_of_tensors:
            tensor.encoder._scale = 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # 恢复原始 scale
        for i, tensor in enumerate(self.list_of_tensors):
            tensor.encoder._scale = self.encodings_cache[i]