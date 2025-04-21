import torch
from torch import nn
import torch.nn.functional as F

def w8_a16_forward(weight, input, scales, bias=None):
    casted_weights = weight.to(input.dtype)
    output = torch.nn.functional.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 bias=True, dtype=torch.float32):
        super().__init__()
        
        
        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        
        self.register_buffer("scales", 
                             torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", 
                                 torch.randn((1, out_features), 
                                             dtype=dtype))
        
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127  # 按列取max，所以是per channel量化
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def forward(self, input):
        return w8_a16_forward(self.int8_weights, 
                              input, self.scales, self.bias)

def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class, module_name_to_exclude)


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_id = "./models/Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
previous_memory_footprint = model.get_memory_footprint()
print("************************* Model before ******************** \n", model)
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
print("************************* Model after ******************** \n", model)
new_memory_footprint = model.get_memory_footprint()
print(f"save memory:{(previous_memory_footprint - new_memory_footprint)/1e+6} MB")

# 保存量化后的模型
# quantized_state_dict = model.state_dict()
# torch.save(quantized_state_dict, "codegen-350M-mono-quant.pth")
