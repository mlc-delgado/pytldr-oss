FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip cmake antiword
RUN CUDACXX=/usr/local/cuda-12.4/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 python3 -m pip install llama-cpp-python==0.2.75 --no-cache-dir --force-reinstall --upgrade
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir