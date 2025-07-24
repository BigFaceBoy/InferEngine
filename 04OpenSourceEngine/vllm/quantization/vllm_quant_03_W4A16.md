# 一、使用compressor量化
```python
from  transformers  import  AutoTokenizer, AutoModelForCausalLM
from  datasets  import  load_dataset
from  llmcompressor.transformers  import  oneshot
from  llmcompressor.modifiers.quantization  import  GPTQModifier
from  llmcompressor.modifiers.smoothquant  import  SmoothQuantModifier
from  vllm  import  LLM

model_path = "./Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
# Load and preprocess the dataset
ds = load_dataset("./ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
def  preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def  tokenize(sample):
    return  tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithms
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model
SAVE_DIR = "./Qwen2.5-7B-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

# 二、使用lm_eval评估
## gsm8k
```
lm_eval --model vllm \
        --model_args pretrained="./Qwen2.5-7B-W4A16",add_bos_token=true  \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 1000
```

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.774|±  |0.0132|
|     |       |strict-match    |     5|exact_match|↑  |0.685|±  |0.0147|


## mmlu
```
lm_eval --model vllm \
        --model_args pretrained="./Qwen2.5-7B-W4A16",add_bos_token=true,tensor_parallel_size=4,gpu_memory_utilization=0.7  \
        --tasks mmlu \
        --num_fewshot 5  \
        --limit 1000    \
        --batch_size 'auto'
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.7420|±  |0.0036|
| - humanities     |      2|none  |      |acc   |↑  |0.6977|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.7647|±  |0.0073|
| - social sciences|      2|none  |      |acc   |↑  |0.8304|±  |0.0066|
| - stem           |      2|none  |      |acc   |↑  |0.6920|±  |0.0080|

# 三、使用vllm推理
```
vllm serve models/Qwen2.5-7B-W4A16  --disable-log-request --host 0.0.0.0 --port 20010
```
# 四、vllm serve + benchmark
```
python code/benchmarks/benchmark_serving.py \
       --backend vllm     \
       --model models/Qwen2.5-7B-W4A16    \
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
Benchmark duration (s):                  54.08
Total input tokens:                      217393
Total generated tokens:                  197759
Request throughput (req/s):              18.49
Output token throughput (tok/s):         3656.61
Total Token throughput (tok/s):          7676.26
---------------Time to First Token----------------
Mean TTFT (ms):                          75.30
Median TTFT (ms):                        55.57
P99 TTFT (ms):                           287.05
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.32
Median TPOT (ms):                        25.53
P99 TPOT (ms):                           29.48
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.69
Median ITL (ms):                         23.84
P99 ITL (ms):                            34.46
==================================================

```