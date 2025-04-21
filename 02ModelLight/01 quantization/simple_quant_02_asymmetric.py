import torch

def linear_dequant(tensor,scale,zero_point):
    #注意这里的tensor进行了强制类型转换，否则容易出现溢出的风险
    r_tensor = (tensor.float() - zero_point) * scale
    return r_tensor

def linear_quant_with_zero_point(tensor,scale,zero_point,dtype=torch.int8):
    scaled_tensor = tensor / scale + zero_point
    round_tensor = torch.round(scaled_tensor)
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    q_tensor = round_tensor.clamp(q_min,q_max).to(dtype)
    return q_tensor

def get_scale_and_zero_point(tensor,dtype=torch.int8):
    q_max = torch.iinfo(dtype).max
    q_min = torch.iinfo(dtype).min
    r_max = tensor.max().item()
    r_min = tensor.min().item()
    scale = (r_max - r_min)/(q_max - q_min)
    zero_point = q_min - (r_min / scale)
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point =int(round(zero_point))
    return scale,zero_point

r = torch.tensor([
    [191.6,-13.5,728.6],
    [92.14,295.5,-184],
    [0,684.6,245.5]
    ])

s,z=get_scale_and_zero_point(r, torch.int8)
print(f"scale:{s} zero_point:{z}")
q = linear_quant_with_zero_point(r, s, z)
print(f"quantized tensor:{q}")

r_dequant = linear_dequant(q, s, z)
print(f"dequantized tensor:{r_dequant}")

quant_error = r - r_dequant
print(f"quant_error tensor:{quant_error}")
print(f"quant_error:{quant_error.square().mean()}")