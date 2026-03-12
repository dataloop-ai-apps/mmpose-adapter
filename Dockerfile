FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
# Base image has torch 2.9.1+cu128, but mmcv 2.2.0 only has pre-built wheels up to torch 2.4.0
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# Base image setuptools 80.9 removed pkg_resources (needed by mmengine)
RUN pip install openmim==0.3.9 "setuptools<78"

USER 1000

RUN pip install "mmengine==0.10.7"
RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4.0/index.html
RUN pip install chumpy --no-build-isolation
RUN pip install "mmdet==3.3.0" "mmpose==1.3.2"

# Patch mmdet to accept mmcv 2.2.0 (upstream cap is <2.2.0)
RUN MMDET_INIT=$(python -c "import importlib.util; print(importlib.util.find_spec('mmdet').submodule_search_locations[0] + '/__init__.py')") \
    && sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET_INIT"

# Base image opencv 4.11 has binary incompatibility with xtcocotools
RUN pip install "opencv-python<4.10" 

# docker build -t gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.7 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/mmpose-adapter:0.2.7
