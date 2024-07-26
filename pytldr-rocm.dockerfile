FROM rocm/dev-ubuntu-22.04:6.1
RUN apt update && apt install -y rocm-hip-sdk rocm-hip-libraries antiword
RUN LLAMA_HIPBLAS=on HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=11.0.0 LLAMA_CUDA_DMMV_X=64 LLAMA_CUDA_MMV_Y=2 CMAKE_ARGS="-DLLAMA_HIPBLAS=on -DAMDGPU_TARGETS=gfx1100" FORCE_CMAKE=1 CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ python3 -m pip install llama-cpp-python==0.2.75 --force-reinstall --upgrade --no-cache-dir
RUN python3 -m pip install --upgrade --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.0
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir