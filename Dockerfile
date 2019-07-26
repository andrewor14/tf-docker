# Ubuntu 18.04 Python3 with CUDA 10 and the following:
#  - Installs tf-nightly-gpu-2.0-preview
#  - Installs requirements.txt for tensorflow/models

FROM nvidia/cuda:10.0-base-ubuntu18.04 as base
ARG tensorflow_pip_spec="tf-nightly-gpu-2.0-preview"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        software-properties-common \
        unzip \
        lsb-core \
        curl

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0 \
        && apt-get clean

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Add google-cloud-sdk to the source list
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install extras needed by most models
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      wget \
      htop \
      zip \
      vim \
      google-cloud-sdk

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv

# Set up Python3 environment
RUN pip3 install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
RUN pip3 install wheel
RUN pip3 install --upgrade setuptools google-api-python-client pyyaml google-cloud google-cloud-bigquery mock
RUN pip3 install absl-py
RUN pip3 install --upgrade --force-reinstall ${tensorflow_pip_spec}
RUN pip3 install tfds-nightly
RUN pip3 install -U scikit-learn
RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 freeze


# =================================
# |          AUTOSCALING          |
# =================================

# Set up working directory
RUN mkdir -p /root/dev
RUN mkdir -p /root/dev/logs
RUN mkdir -p /root/dev/train_data
RUN mkdir -p /root/dev/eval_data
RUN mkdir -p /root/dev/dataset
RUN mkdir -p /root/lib

# Install openmpi
WORKDIR /root/lib
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz
RUN tar -vzxf openmpi-4.0.1.tar.gz
WORKDIR openmpi-4.0.1
RUN ./configure --with-cuda=/usr/local/cuda
RUN make all install
ENV MPI_HOME=/usr/local

# Clone various repos
WORKDIR /root/dev
RUN git clone -b autoscaling https://github.com/andrewor14/models
RUN git clone -b autoscaling https://github.com/andrewor14/tensorflow
RUN git clone --recursive https://github.com/andrewor14/horovod

# Fetch datasets
WORKDIR /root/dev/dataset
RUN mkdir cifar10
WORKDIR cifar10
RUN wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
RUN tar -vzxf cifar-10-binary.tar.gz

