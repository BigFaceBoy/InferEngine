# 一、使用 autoawq 量化
awq 会默认使用 mit-han-lab/pileval 的校准数据集，如果代码无法直接访问，可事先下载huggingface上的数据集，然后将安装的awq中的dataset路径改为自己环境的路径：
文件路径为：envs/xxx/lib/python3.xx/site-packages/awq/utils/calib_data.py ，如修改为：
```
dataset = load_dataset("/data/home/fangxuwei/Code/AutoAWQ/mit-han-lab", split="validation")
```

```python
from  awq  import  AutoAWQForCausalLM
from  transformers  import  AutoTokenizer

model_path = "./Qwen2.5-7B"
quant_path = "./Qwen2.5-7B-AWQ"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')
```
这里出现问题：TypeError: Qwen2Attention.forward() missing 1 required positional argument: 'attention_mask'
解决方式：当前transformers版本是4.51.3，安装 transformers==4.47.1 可以解决。

# 二、使用lm_eval评估
## gsm8k
```
lm_eval --model vllm  \
        --model_args pretrained="./Qwen2.5-7B-AWQ/",add_bos_token=true,quantization="AWQ",dtype="half" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 250 \
        --batch_size 'auto'
```
输出
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.814|±  |0.0123|
|     |       |strict-match    |     5|exact_match|↑  |0.740|±  |0.0139|


```

## mmlu
```
lm_eval --model vllm \
        --model_args pretrained="./Qwen2.5-7B-AWQ",add_bos_token=true,,quantization="AWQ",dtype="half",tensor_parallel_size=4,gpu_memory_utilization=0.7 \
        --tasks mmlu \
        --num_fewshot 5  \
        --limit 1000    \
        --batch_size 'auto'

```

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.7398|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.6869|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.7721|±  |0.0073|
| - social sciences|      2|none  |      |acc   |↑  |0.8268|±  |0.0067|
| - stem           |      2|none  |      |acc   |↑  |0.6930|±  |0.0080|

# 三、使用vllm推理
```python
from  vllm  import  LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="./Qwen2.5-7B-AWQ", quantization="AWQ", dtype="half")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

# 四、使用vllm benchmark
## 4.1 使用vllm serve 部署服务
```
vllm  serve Qwen2.5-7B-AWQ --disable-log-requests --quantization awq --dtype="half"
```
## 4.2 benchmark
```
python /vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Qwen2.5-7B-AWQ \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json  \
    --num-prompts 1000 \
    --host 0.0.0.0 \
    --port 20010 \
    --max-concurrency 100
```

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  162.11
Total input tokens:                      217393
Total generated tokens:                  198058
Request throughput (req/s):              6.17
Output token throughput (tok/s):         1221.73
Total Token throughput (tok/s):          2562.73
---------------Time to First Token----------------
Mean TTFT (ms):                          159.89
Median TTFT (ms):                        139.78
P99 TTFT (ms):                           395.61
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          69.52
Median TPOT (ms):                        69.67
P99 TPOT (ms):                           73.38
---------------Inter-token Latency----------------
Mean ITL (ms):                           69.04
Median ITL (ms):                         68.28
P99 ITL (ms):                            77.94
==================================================

```

llmcompressor
```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  82.36
Total input tokens:                      217393
Total generated tokens:                  197656
Request throughput (req/s):              12.14
Output token throughput (tok/s):         2399.80
Total Token throughput (tok/s):          5039.23
---------------Time to First Token----------------
Mean TTFT (ms):                          91.18
Median TTFT (ms):                        74.45
P99 TTFT (ms):                           279.95
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.63
Median TPOT (ms):                        36.67
P99 TPOT (ms):                           40.63
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.09
Median ITL (ms):                         34.75
P99 ITL (ms):                            45.68
==================================================
```