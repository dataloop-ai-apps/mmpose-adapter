FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libgl1-mesa-glx libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pin PyTorch 2.3.0 + CUDA 12.1 (required for mmcv 2.2.0 pre-built wheel availability)
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install openmim==0.3.9 "setuptools<78"

USER 1000

# Install OpenMMLab stack with pinned compatible versions
RUN mim install "mmengine==0.10.7"
RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
RUN pip install chumpy --no-build-isolation
RUN pip install "mmdet==3.3.0" "mmpose==1.3.2"

# Patch mmdet to accept mmcv 2.2.0 (upstream bug: github.com/open-mmlab/mmdetection/issues/12312)
RUN python -c "\
import mmdet, os; \
p = os.path.join(os.path.dirname(mmdet.__file__), '__init__.py'); \
c = open(p).read().replace(\"mmcv_maximum_version = '2.2.0'\", \"mmcv_maximum_version = '2.3.0'\"); \
open(p, 'w').write(c)"

# Pin numpy<2 and opencv<4.10 for binary compatibility with xtcocotools/pycocotools
RUN pip install "numpy<2" "opencv-python<4.10"

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.7 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.7