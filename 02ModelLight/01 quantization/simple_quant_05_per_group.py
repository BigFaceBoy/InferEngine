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

def get_scale_symmetric(tensor,dtype=torch.int8):
    q_max = torch.iinfo(dtype).max
    r_max = tensor.max().item()
    scale = r_max/q_max
    return scale

def linear_quant_per_channel(tensor, dim, dtype=torch.int8):
    output_dim = tensor.shape[dim] # 沿dim的维度大小
    scale = torch.zeros(output_dim) # 创建一个对应维度的tensor，用于存储scale
    for i in range(output_dim):
        sub_tensor = tensor.select(dim, i)
        scale[i] = get_scale_symmetric(sub_tensor, dtype=dtype)
    
    scale_shape = [1] * tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quant_with_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)
    return quantized_tensor, scale


def linear_quant_per_group(tensor, group_size, dtype=torch.int8):
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2

    tensor = tensor.view(-1, group_size)
    print(tensor)
    quantized_tensor, scale = linear_quant_per_channel(tensor, dim=0, dtype=dtype)
    quantized_tensor = quantized_tensor.view(t_shape)

    return quantized_tensor, scale

def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)
    dequantized_tensor = linear_dequant(quantized_tensor, scale, 0)
    dequantized_tensor = dequantized_tensor.view(q_shape)
    return dequantized_tensor

r = torch.tensor([
    [0.61, 0.30, 0.09, 0.11, 0.35, 0.22],
    [0.81, 0.22, 0.70, 0.39, 0.85, 0.21],
    [0.98, 0.82, 0.20, 0.95, 0.66, 0.69],
    [0.95, 0.73, 0.79, 0.19, 0.50, 0.53],
    [0.21, 0.94, 0.91, 0.39, 0.86, 0.63],
    [0.90, 0.71, 0.20, 0.33, 0.72, 0.84]
    ])

group_size = 3
q, s= linear_quant_per_group(r, group_size)
print(f"quantized tensor:{q}")

r_dequant = linear_dequantization_per_group(q, s, group_size)
print(f"dequantized tensor:{r_dequant}")

quant_error = r - r_dequant
print(f"quant_error tensor:{quant_error}")
print(f"quant_error:{quant_error.square().mean()}")  # 2.15