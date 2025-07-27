FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.devel.py3.8.pytorch2

USER root
# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libgl1-mesa-glx libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install MMEngine and MMCV
RUN pip install openmim==0.3.9


USER 1000

RUN mim install "mmengine>=0.10.7" "mmcv==2.0.0rc4" "mmdet==3.0.0" "mmpose==1.0.0"
RUN pip install numpy --upgrade

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.6 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.6