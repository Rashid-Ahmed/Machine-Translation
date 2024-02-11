# Use the nvidia/cuda base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.9.6
RUN curl -O https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz \
    && tar -xf Python-3.9.6.tgz \
    && cd Python-3.9.6 \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-3.9.6 Python-3.9.6.tgz

# Update alternatives to use Python 3.9.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1

# Install pip for Python 3.9.6
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Upgrade pip and install Poetry
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install poetry
