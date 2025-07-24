以下所有操作均在nvidia l40 上进行。

## 一、安装
### full build
```
git  clone https://github.com/vllm-project/vllm.git
python  use_existing_torch.py
pip  install -r requirements/build.txt
export MAX_JOBS=10
pip  install -e . --no-build-isolation
```
注意，需要当前编译器支持c+\+17。由于宿主机是gcc 4.8.5, 所以从源码编译了个gcc 8.5.0, 然后加到环境变量里。设置c+\+编译器， export CXX=/path/to/gcc-8.5.0/bin/c+\+
在CMakeLists.txt 中增加头文件路径：
include_directories(
     /path-to-gcc/include/c++/8.5.0/
)
### build python only
如果只修改了python代码，可以使用pre-built whl 节省编译时间：
```
export VLLM_PRECOMPILED_WHEEL_LOCATION=dist/vllm-xxx.whl
python setup.py bdist_wheel
```
## 二、使用vllm
```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="../models/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 三、vllm serve
### 3.1 启动服务
```
vllm serve Qwen2.5-7B --disable-log-requests --host 0.0.0.0 --port 20010
```
### 3.2 benchmark
```
python /path-to-vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Qwen2.5-7B \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path /path-to-datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json  \
    --num-prompts 1000 \
    --host 0.0.0.0 \
    --port 20010 \
    --max-concurrency 100
```
输出：
```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  81.56
Total input tokens:                      217393
Total generated tokens:                  197640
Request throughput (req/s):              12.26
Output token throughput (tok/s):         2423.37
Total Token throughput (tok/s):          5088.94
---------------Time to First Token----------------
Mean TTFT (ms):                          91.53
Median TTFT (ms):                        74.11
P99 TTFT (ms):                           310.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.22
Median TPOT (ms):                        36.24
P99 TPOT (ms):                           40.67
---------------Inter-token Latency----------------
Mean ITL (ms):                           35.72
Median ITL (ms):                         34.43
P99 ITL (ms):                            45.14
==================================================

```
