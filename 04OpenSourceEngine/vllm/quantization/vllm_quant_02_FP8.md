# 一、使用compressor量化
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from vllm import LLM

model_path = "./Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


# Configure the quantization algorithms
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
# Apply quantization
oneshot(
    model=model,
    recipe=recipe
)

# Save the compressed model
SAVE_DIR = "./Qwen2.5-7B-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

```
# 二、使用lm_eval评估
## gsm8k
```
lm_eval --model vllm \
        --model_args pretrained="./Qwen2.5-7B-FP8-Dynamic/",add_bos_token=true \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 1000 \
        --batch_size 'auto'
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.841|±  |0.0116|
|     |       |strict-match    |     5|exact_match|↑  |0.793|±  |0.0128|

## mmlu
```
lm_eval --model vllm \
        --model_args pretrained="./Qwen2.5-7B-FP8-Dynamic",add_bos_token=true,tensor_parallel_size=4,gpu_memory_utilization=0.7  \
        --tasks mmlu \
        --num_fewshot 5  \
        --limit 1000   \
        --batch_size 'auto'
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.7479|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.6974|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.7715|±  |0.0073|
| - social sciences|      2|none  |      |acc   |↑  |0.8372|±  |0.0065|
| - stem           |      2|none  |      |acc   |↑  |0.7044|±  |0.0079|

# 三、使用vllm推理
```
vllm serve models/Qwen2.5-7B-FP8-Dynamic  --disable-log-request --host 0.0.0.0 --port 20010
```
# 四、vllm serve + benchmark
```
python code/benchmarks/benchmark_serving.py \
       --backend vllm     \
       --model models/Qwen2.5-7B-FP8-Dynamic    \
       --endpoint /v1/completions \
       --dataset-name sharegpt    \
       --dataset-path datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json  \
       --num-prompts 1000  \
       --host 0.0.0.0   \
       --port 20010  \
       --max-concurrency 100
```


```

============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  57.09
Total input tokens:                      217393
Total generated tokens:                  197767
Request throughput (req/s):              17.52
Output token throughput (tok/s):         3463.91
Total Token throughput (tok/s):          7271.58
---------------Time to First Token----------------
Mean TTFT (ms):                          73.69
Median TTFT (ms):                        55.71
P99 TTFT (ms):                           284.29
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.65
Median TPOT (ms):                        25.77
P99 TPOT (ms):                           29.17
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.19
Median ITL (ms):                         23.79
P99 ITL (ms):                            33.93
==================================================

```

