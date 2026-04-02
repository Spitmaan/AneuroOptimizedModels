# Guide for PC General
Since the Jetson Orin Nano Super has 8GB of RAM, building a TensorRT Edge-LLM engine directly on the device often causes Out-of-Memory (OOM) errors, as the process requires roughly 4x the model's memory footprint. To avoid this, you should export the model on an x86 host (your PC) and then build the final engine on the Jetson using specific memory-saving flags. [1, 2]

## 1. Prepare the x86 Host (PC)

The first phase happens on your workstation to convert the raw model (e.g., from Hugging Face) into a format the Edge-LLM runtime understands. [3]

-   Clone the Repo: Get the [TensorRT-Edge-LLM repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) on both your PC and your Jetson.
-   Install Python Dependencies: On your PC, set up a virtual environment and install the required tools:
    
    ```bash
    pip install tensorrt_edge_llm
    
    ```
    
-   Export the Model: Use the provided scripts to convert your model. For the Orin Nano Super, INT4 quantization is highly recommended to ensure the model fits in the 8GB RAM.
    
    ```bash
    # Example for a Llama-3-8B model
    python3 export.py --model_dir ./llama-3-8b --quantization int4 --output_dir ./exported_model
    
    ```
    
    [1, 4, 5]

## 2. Build the Engine on Jetson [1]

Once exported, move the `./exported_model` folder to your Jetson Orin Nano Super. Although this is technically "on-device," we call it the deployment phase because the heavy "export" lifting was done on the PC.

-   Build the C++ Runtime: On the Jetson, navigate to the repo and build the project:
    
    ```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    ```
    
-   Generate the Engine: Run the engine builder. To prevent crashes on the 8GB Orin Nano Super, use the `--mmap` flag if available in your version to manage memory usage more efficiently.
-   Enable "Super Mode": Before running inference, ensure your Jetson is in its highest performance state using JetPack 6.2:
    
    ```bash
    sudo nvpmodel -m 0  # Sets to MAXN SUPER mode
    sudo jetson_clocks  # Locks clocks to maximum frequency
    
    ```
    
    [1, 6, 7]

## 3. Alternative: Use Pre-built Containers

If cross-compiling feels too complex, the jetson-containers project provides pre-compiled environments. [8]

1.  Clone the tool: `git clone https://github.com/dusty-nv/jetson-containers`.
2.  Run the container: `./run.sh $(./autotag tensorrt_llm)` to launch an environment that already has the necessary libraries matched to your JetPack version.

## Recommended Models for Orin Nano Super (8GB)

-   Llama-3-8B (INT4): The upper limit for this device; requires careful memory management.
-   Gemma-2B / Phi-3 Mini: Runs very comfortably with high tokens-per-second.
-   VILA-2.7B: Excellent for vision-language tasks (VLMs) at the edge. [7]

Would you like the exact shell commands for a specific model like Llama-3 or Phi-3?

  

[1] [https://github.com](https://github.com/NVIDIA/TensorRT-LLM/issues/3149)

[2] [https://www.youtube.com](https://www.youtube.com/watch?v=S8yHig1e-Ks)

[3] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html)

[4] [https://www.youtube.com](https://www.youtube.com/watch?v=PY41CP13p3k&t=4)

[5] [https://www.youtube.com](https://www.youtube.com/watch?v=3LWgrN3m2pY&t=308)

[6] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html)

[7] [https://www.youtube.com](https://www.youtube.com/watch?v=TSbl5ZxdbPk)

[8] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/clear-guide-to-install-all-the-ai-model-training-components/359734)

--------
-------

# Guide for DGX
Setting up TensorRT Edge-LLM on the NVIDIA DGX Spark is a streamlined process because the [DGX OS](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/software-requirements.html) comes pre-loaded with the necessary NVIDIA AI stack, including CUDA, cuDNN, and the [TensorRT SDK](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html). [1, 2]

Since the DGX Spark is an ARM64 system (Grace Blackwell), you will follow the same steps used for a high-performance host to export models for your Jetson Orin Nano Super. [3]

## 1. Initial Environment Setup

Open a terminal on your DGX Spark and create a dedicated Python virtual environment to keep your system clean.

```bash
# Update package list and install pip if necessary
sudo apt update && sudo apt install python3-pip python3-venv -y

# Create and activate a virtual environment
python3 -m venv ~/edge_llm_env
source ~/edge_llm_env/bin/activate

# Upgrade essential build tools
pip install --upgrade pip setuptools wheel

```

## 2. Install TensorRT Edge-LLM

You can install the package directly from NVIDIA's repositories. Since you are on a DGX Spark (Ubuntu 24.04 ARM64), ensure you are using the versions compatible with your [CUDA 13.x installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html). [4, 5, 6, 7, 8]

```bash
# Install the core Edge-LLM package
pip install tensorrt_edge_llm

# (Optional) Clone the repository for access to example export scripts
git clone https://github.com
cd TensorRT-Edge-LLM
pip install -r requirements.txt

```

## 3. Verify the Installation

To ensure the DGX Spark's Blackwell GPU is correctly recognized by the software:

```python
# Run this in a python shell
import tensorrt_edge_llm
import tensorrt as trt

print(f"TensorRT Edge-LLM version: {tensorrt_edge_llm.__version__}")
print(f"TensorRT SDK version: {trt.__version__}")

```

## 4. Running the Export Workflow

Now that your DGX Spark is set up, you can use its 128GB of unified memory to perform the heavy quantization and export required for your Orin Nano Super. [9, 10]

-   Download Model: Use `huggingface-cli` to download your target model (e.g., Llama-3).
-   Run Export: Use the `export.py` script provided in the repository to generate the model checkpoint.
-   Transfer: Copy the resulting `./exported_model` folder to your Jetson Orin Nano Super via SFTP or a USB drive to build the final engine on the target device. [11]

## Pro-Tip: Using Containers

If you prefer a completely isolated environment, you can use the official TensorRT-LLM container from the NVIDIA NGC Catalog. [12, 13]

```bash
# Pull and run the official development container
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

```

Would you like the exact Python script to export a specific model like Llama-3-8B on your DGX Spark?

  

[1] [https://docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/software-requirements.html)

[2] [https://signal65.com](https://signal65.com/research/nvidia-dgx-spark-first-look-a-personal-ai-supercomputer-on-your-desk/)

[3] [https://docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)

[4] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html)

[5] [https://mintlify.com](https://mintlify.com/NVIDIA/TensorRT-LLM/installation)

[6] [https://www.youtube.com](https://www.youtube.com/watch?v=CIqV8z_fmVE&t=105)

[7] [https://build.nvidia.com](https://build.nvidia.com/spark/trt-llm)

[8] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/setup-tensorrt-in-drive-orin/347699)

[9] [https://www.nvidia.com](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

[10] [https://docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

[11] [https://build.nvidia.com](https://build.nvidia.com/spark/trt-llm)

[12] [https://github.com](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/linux.md)

[13] [https://github.com](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/build-from-source-linux.md)


# Example for Llama-3-8B

To export Llama-3-8B on your DGX Spark for deployment on a Jetson Orin Nano Super, you will use the `export.py` script provided in the [TensorRT Edge-LLM repository](https://github.com/NVIDIA/TensorRT-Edge-LLM).

Since your Jetson has 8GB of RAM, it is critical to use INT4 quantization during this export stage to ensure the model fits within the device's memory limits. [1, 2]

## 1. Download the Model

First, ensure you have the model weights on your DGX Spark. You can use the Hugging Face CLI to download them: [3]

```bash
# Authenticate with your Hugging Face token
huggingface-cli login

# Download Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./llama-3-8b

```

## 2. Run the Export Script

Navigate to your cloned `TensorRT-Edge-LLM` directory and run the following command. The DGX Spark's 128GB of memory will handle this process easily. [4, 5]

```bash
python3 export.py \
    --model_dir ./llama-3-8b \
    --quantization int4_awq \
    --output_dir ./exported_llama3_int4 \
    --dtype float16 \
    --max_input_len 2048 \
    --max_output_len 512

```

## 3. Key Parameters Explained

-   `--quantization int4_awq`: This is the most important flag for your 8GB Jetson. It reduces the model size by roughly 75%, allowing it to run smoothly.
-   `--dtype float16`: Sets the base precision for activations, which is well-supported by the Orin architecture.
-   `--max_input_len` / `--max_output_len`: Defines the sequence limits. Keeping these conservative (e.g., 2048/512) helps save KV cache memory on your Jetson. [6, 7, 8]

## 4. Next Step: Transfer to Jetson

Once the script finishes, you will see a new folder named `exported_llama3_int4`. Copy this entire folder to your Jetson Orin Nano Super:

```bash
# Example using SCP to move the model to your Jetson
scp -r ./exported_llama3_int4 <jetson_user>@<jetson_ip>:~/

```

On the Jetson, you will then use the C++ builder included in the Edge-LLM runtime to create the final `.engine` file specifically for the Orin Nano hardware. [5, 9]

Would you like the C++ build commands to run on your Jetson once the files are transferred?

  

[1] [https://github.com](https://github.com/NVIDIA/TensorRT-LLM/issues/3149#:~:text=As%20we%20discussed%20below%2C%20you%20can%20use,you%20can%20use%20%2Dmmap%2C%20pls%20refer%20https://github.com/NVIDIA/TensorRT%2DLLM/blob/v0.12.0%2Djetson/README4Jetson.md%233%2Dreference%2Dmemory%2Dusage.)

[2] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412#:~:text=Model%20Format%20&%20Sizing.%20By%20using%20the,~10B%20parameters.%20VLMs%20up%20to%20~4B%20parameters.)

[3] [https://www.hackster.io](https://www.hackster.io/shahizat/running-llms-with-tensorrt-llm-on-nvidia-jetson-agx-orin-34372f)

[4] [https://github.com](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md)

[5] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html)

[6] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/tensorrt-quantization-for-jetson-inference/347929)

[7] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227)

[8] [https://developer.nvidia.com](https://developer.nvidia.com/blog/deploy-an-ai-coding-assistant-with-nvidia-tensorrt-llm-and-nvidia-triton/)

[9] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html#:~:text=TensorRT%20Edge%2DLLM%20has%20two%20separate%20components%20that,Runtime%20%28builds%20and%20runs%20on%20Edge%20devices%29)

# C++ Buil In Jetson Orin Nano Super
After transferring the `exported_llama3_int4` folder from your DGX Spark to your Jetson Orin Nano Super, you need to build the C++ runtime and then use the Engine Builder to create the final optimized engine file.

## 1. Build the C++ Runtime

On your Jetson, navigate to your [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) directory and compile the runtime libraries. Use the `-j$(nproc)` flag to use all CPU cores for a faster build. [1]

```bash
cd TensorRT-Edge-LLM
mkdir -p build && cd build

# Configure the build for the Jetson Orin architecture
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMBEDDED_TARGET=jetson-orin

# Compile the runtime (takes 1-2 minutes)
make -j$(nproc)

```

## 2. Build the TensorRT Engine

Now, use the compiled `engine_builder` tool to create the final `.engine` file from the folder you transferred. Ensure you provide the absolute path to your exported model folder. [2, 3]

```bash
# From the 'build' directory created above
./examples/engine_builder/engine_builder \
    --model_dir ~/exported_llama3_int4 \
    --output_file ~/llama3_int4.engine \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_output_len 512

```

## 3. Run Inference

Once the engine is built, you can run a quick test using the included chat example to verify the model is responding correctly: [4]

```bash
./examples/chat/chat \
    --engine_file ~/llama3_int4.engine \
    --tokenizer_dir ~/exported_llama3_int4/tokenizer

```

## Critical Performance Tips for 8GB RAM

-   Use Swap: If the engine builder crashes, it is likely due to an OOM (Out of Memory) error. You can temporarily increase your swap size on the Jetson to help it finish the build.
-   Super Mode: Always ensure your Jetson is in MAXN SUPER mode via the `nvpmodel` tool before building or running inference to maximize the available 67 TOPS.
-   KV Cache Management: If you experience slow responses, try reducing the `--max_input_len` to `1024` to free up more memory for the model weights. [5]

Would you like the Python script to run a simple performance benchmark on your new engine?

  

[1] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html)

[2] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html)

[3] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html)

[4] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html#:~:text=Prerequisites.%20Step%201:%20Clone%20Repository%20%28on%20both,device%29%20Step%206:%20Run%20Inference.%20Next%20Steps.)

[5] [https://developer.nvidia.com](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/)

# Potential Benchmarking
To benchmark your Llama-3-8B engine on the Jetson Orin Nano Super, you can use this Python script which utilizes the `time` and `tensorrt_edge_llm` libraries to measure Tokens Per Second (TPS) and Time to First Token (TTFT).

## Performance Benchmark Script

```python
import time
import tensorrt_edge_llm
from transformers import AutoTokenizer

# --- CONFIGURATION ---
ENGINE_PATH = "/home/jetson/llama3_int4.engine"
TOKENIZER_DIR = "/home/jetson/exported_llama3_int4/tokenizer"
PROMPT = "Explain the importance of physical AI in 50 words."

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").squeeze().tolist()

# Initialize Runtime
runtime = tensorrt_edge_llm.LLMRuntime(ENGINE_PATH)

print(f"--- Starting Benchmark on Jetson Orin Nano Super ---")
print(f"Prompt: {PROMPT}\n")

# Start Timing
start_time = time.perf_counter()
first_token_time = None
token_count = 0

# Inference Loop
for output_token in runtime.generate(input_ids, max_new_tokens=128):
    if first_token_time is None:
        first_token_time = time.perf_counter()
    token_count += 1
    # print(tokenizer.decode([output_token]), end="", flush=True) # Uncomment to see text

end_time = time.perf_counter()

# --- RESULTS ---
total_time = end_time - start_time
ttft = (first_token_time - start_time) * 1000 # ms
tps = token_count / (end_time - first_token_time)

print(f"\n\n--- Benchmark Results ---")
print(f"Tokens Generated: {token_count}")
print(f"Time to First Token (TTFT): {ttft:.2f} ms")
print(f"Tokens Per Second (TPS): {tps:.2f} tok/s")
print(f"Total Inference Time: {total_time:.2f} s")

```

## Pre-Benchmark Checklist

Before running the script, ensure your Jetson is in its highest performance state to get accurate results: [1, 2]

1.  Set Power Mode: `sudo nvpmodel -m 0` (MAXN SUPER mode).
2.  Lock Clocks: `sudo jetson_clocks`.
3.  Monitor Resources: Open a second terminal and run `sudo tegrastats` to watch memory and GPU utilization in real-time. [3, 4, 5]

## Expected Performance

On an Orin Nano Super (8GB) with INT4 quantization, you should typically see:

-   Throughput: Between 10–20 tokens/second for Llama-3-8B.
-   Memory Usage: Approximately 5.5GB–6.5GB of shared RAM (leaving enough room for the OS and small background tasks). [6]

Would you like to know how to further optimize these results using speculative decoding?

  

[1] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/tensorrt-llm-on-jetson-orin-nx-16gb/318484)

[2] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/jetpack6-llamacpppython/318634)

[3] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/benchmark-test-for-jetson-orin-nano/361889)

[4] [https://www.cytron.io](https://www.cytron.io/tutorial/building-rag-system-nvidia-orin-nano-local)

[5] [https://www.jetson-ai-lab.com](https://www.jetson-ai-lab.com/tutorials/genai-benchmarking/)

[6] [https://www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/1rqn8q1/running_8b_llama_locally_on_jetson_orin_nano_with/#:~:text=Running%208B%20Llama%20locally%20on%20Jetson%20Orin,comparison%20against%20a%20llama.%20cpp%20INT4%20baseline.)

# Speculative Engine Example
To further optimize your Llama-3-8B performance on the Jetson Orin Nano Super, you can use speculative decoding. This technique uses a smaller "draft" model (like a 1B or 2B model) to predict multiple tokens at once, which the larger 8B "target" model then verifies in a single pass. [1, 2, 3]

For the most advanced results, NVIDIA TensorRT Edge-LLM supports EAGLE-3, which uses a lightweight prediction head instead of a separate draft model to achieve up to 3x speedups. [4, 5]

## 1. Build the Speculative Engine

On your DGX Spark, you must rebuild the target engine to support speculative tokens by adding specific flags to the `trtllm-build` command. [6, 7]

```bash
# On DGX Spark: Build target engine with speculative support
trtllm-build \
    --checkpoint_dir ./ckpt-target-8b \
    --output_dir ./target-engine \
    --speculative_decoding_mode draft_tokens_external \
    --max_draft_len 5 \
    --use_paged_context_fmha enable

```

## 2. Prepare the Draft Model

You will also need a smaller draft model (e.g., Llama-3.2-1B) exported in the same way as your 8B model. Ensure it is also quantized to INT4 to fit alongside the main model in the Jetson's 8GB memory. [8]

## 3. Updated Benchmark Script (Speculative)

Transfer both the target and draft engines to your Jetson. Use this updated Python snippet to initialize the speculative runtime.

```python
import tensorrt_edge_llm

# Load the target and draft engines
runtime = tensorrt_edge_llm.LLMRuntime(
    engine_path="/home/jetson/llama3_8b_target.engine",
    draft_engine_path="/home/jetson/llama3_1b_draft.engine" # Add draft engine
)

# Benchmark as before
for output_token in runtime.generate(input_ids, max_new_tokens=128):
    # Performance is boosted by accepting multiple draft tokens per step
    pass

```

## 4. Hardware Optimization (JetPack 6.2+)

Ensure your Jetson Orin Nano Super is in MAXN SUPER mode (mode 0) to unlock the full 67 TOPS required for the increased parallel compute of speculative decoding. [9, 10]

-   Set Mode: `sudo nvpmodel -m 0`
-   Max Clocks: `sudo jetson_clocks` [9]

## Summary of Expected Gains

Method

Estimated TPS

Latency (TTFT)

Standard (INT4)

~15 tok/s

~50ms

Speculative (EAGLE-3)

~35-45 tok/s

~40ms

Would you like the EAGLE-3 specific export commands to avoid needing a separate draft model?

  

[1] [https://developer.nvidia.com](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

[2] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/345286)

[3] [https://medium.com](https://medium.com/data-science/boosting-llm-inference-speed-using-speculative-decoding-0cb0bf36d001)

[4] [https://developer.nvidia.com](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

[5] [https://docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Speculative_Decoding/TRT-LLM/README.html)

[6] [https://developer.nvidia.com](https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/#:~:text=%23%20Build%20draft%20and%20target%20engines%23%20Important,%2D%2Dmax_draft_len%20must%20be%20specified%20for%20target%20model.)

[7] [https://developer.nvidia.com](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)

[8] [https://rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html)

[9] [https://forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-performance-test-issue/331700)

[10] [https://developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/#:~:text=With%20this%20performance%20boost%2C%20the%20Jetson%20Orin,8B%20parameters%2C%20such%20as%20the%20Llama%2D3.1%2D8B%20model.)

# EAGLE-3 Specific Export Commands
To export Llama-3-8B with EAGLE-3 speculative decoding, you will perform a dual export on your DGX Spark: once for the base model and once for the speculative "drafter" head. EAGLE-3 is highly optimized for edge devices like the Jetson Orin Nano Super because it uses a lightweight prediction head instead of a full secondary model. [1, 2]

## 1. Export the Base Model (INT4) [3]

First, export your Llama-3-8B model with quantization as you did before, but ensure it is ready to be linked with a speculative head. [4, 5]

```bash
# On DGX Spark
python3 export.py \
    --model_dir ./llama-3-8b \
    --quantization int4_awq \
    --output_dir ./llama3_base_int4 \
    --dtype float16

```

## 2. Export the EAGLE-3 Drafter Head [6]

EAGLE-3 requires its own specialized checkpoint that "plugs into" the base model's internal layers. You can download pre-trained EAGLE heads from Hugging Face (e.g., yuhuili/EAGLE-LLaMA3-Instruct-8B). [7, 8, 9]

```bash
# On DGX Spark: Export the speculative head
python3 export.py \
    --model_dir ./eagle-3-llama3-head \
    --output_dir ./llama3_eagle_head \
    --speculative_mode eagle3 \
    --dtype float16

```

## 3. Build the Speculative Engine on Jetson

After transferring both folders to your Jetson, use the `engine_builder` to fuse them into a single high-performance engine. [5, 10]

```bash
# On Jetson Orin Nano Super
./examples/engine_builder/engine_builder \
    --model_dir ~/llama3_base_int4 \
    --speculative_model_dir ~/llama3_eagle_head \
    --output_file ~/llama3_speculative.engine \
    --max_draft_len 4 \
    --max_batch_size 1

```

## 4. Why Use EAGLE-3 on Orin Nano Super?

-   Memory Efficiency: Unlike standard speculative decoding which requires a second model (e.g., a 1B model), the EAGLE-3 head is a tiny fraction of the size, preserving your 8GB of RAM.
-   Massive Speedup: It can deliver 2x to 3x throughput gains, potentially moving your Llama-3-8B performance from ~15 TPS to ~40+ TPS.
-   Single-Pass Verification: It predicts future tokens based on hidden state features rather than just word probabilities, leading to higher "acceptance rates" (more correct guesses per step). [1, 2, 7, 11]

Pro Tip: To maximize these gains, ensure you are in MAXN SUPER mode (`sudo nvpmodel -m 0`) to provide enough compute for the speculative verification cycles. [12]

Would you like to see a comparison of memory usage between standard INT4 and speculative INT4 on your 8GB device?

  

[1] [https://developer.nvidia.com](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

[2] [https://developer.nvidia.com](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

[3] [https://developer.nvidia.com](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms/)

[4] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)

[5] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)

[6] [https://x.com](https://x.com/seeedstudio/status/1782327799413706765)

[7] [https://docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Speculative_Decoding/TRT-LLM/README.html)

[8] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)

[9] [https://docs.vllm.ai](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/)

[10] [https://nvidia.github.io](https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html#:~:text=Prerequisites.%20Step%201:%20Clone%20Repository%20%28on%20both,device%29%20Step%206:%20Run%20Inference.%20Next%20Steps.)

[11] [https://developer.nvidia.com](https://developer.nvidia.com/blog/unlock-faster-smarter-edge-models-with-7x-gen-ai-performance-on-nvidia-jetson-agx-thor/)

[12] [https://www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/1oo88sq/nvidia_jetson_orin_nano_super_8_gb_llamabench/#:~:text=Super%20Power%20Mode%20%28profile%202%29%20enabled%20jwest33@jwest33%2Ddesktop:~/Desktop/llama.cpp$,336.04%20%C2%B1%2014.27%20%7C%20build:%20961660b8c%20%286912%29)