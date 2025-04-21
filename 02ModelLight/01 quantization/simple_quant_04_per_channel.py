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
    
    # 调整 scale 的shape以适配 tensor / scale
    scale_shape = [1] * tensor.dim() # 创建一个长度与tensor维度数相同的列表scale_shape，所有元素初始化为1
    scale_shape[dim] = -1  # 将scale_shape中索引为dim的位置的值改为-1
    scale = scale.view(scale_shape) # 调整scale的形状为scale_shape，使其在dim维度保留原始长度，其他维度为1
    quantized_tensor = linear_quant_with_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)
    return quantized_tensor, scale

r = torch.tensor([
    [191.6, -13.5, 728.6,  452.1],
    [92.14, 295.5, -184, -0.23],
    [0,     684.6, 245.5, 32.4]
    ])


q, s= linear_quant_per_channel(r, 0)
print(f"quantized tensor:{q}")

r_dequant = linear_dequant(q, s, zero_point=0)
print(f"dequantized tensor:{r_dequant}")

quant_error = r - r_dequant
print(f"quant_error tensor:{quant_error}")
print(f"quant_error:{quant_error.square().mean()}")