# Dockerfile for setting up a SUMO environment with reinforcement learning capabilities

FROM ubuntu:22.04

# Set the terminal to non-interactive mode to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive


# Download and install additional dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libxerces-c-dev \
    libfox-1.6-dev \
    libgdal-dev \
    libproj-dev \
    libgl2ps-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    nano \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install SUMO (Simulation of Urban MObility) from the official PPA
RUN git clone https://github.com/eclipse-sumo/sumo.git /opt/sumo && \
    cd /opt/sumo && \
    mkdir build && cd build && \
    cmake .. \
        -DENABLE_GUI=ON \
        -DFOX_CONFIG=/usr/bin/fox-config && \
    make -j$(nproc) && \
    make install

# Set SUMO_HOME environment variable to the default installation path of SUMO in the container
ENV SUMO_HOME=/opt/sumo
# Add SUMO binaries and tools to the PATH environment variable for easy access
ENV PATH="/usr/local/bin:/opt/sumo/bin:${PATH}"


# Install the required Python packages for RL training
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    gymnasium \
    stable-baselines3 \
    torch \
    torchvision \
    tensorboard \
    traci 



# Set the working directory for the container (if does not exist, it will be created)
WORKDIR /workspace