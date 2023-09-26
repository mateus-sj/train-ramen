# Base image with CUDA support
FROM nvcr.io/nvidia/pytorch:22.03-py3

# Set noninteractive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python3.9 \
    pip \
    wget \
    git

# Set the working directory
WORKDIR /work

# Copy the requirements file into the container
COPY requirements.txt /work/

# Install Python dependencies
RUN pip install --no-cache-dir -r /work/requirements.txt

# Copy the rest of the project files into the container
COPY . /work/

# Initialize and update submodules
RUN git submodule update --init

# Set environment variables for CUDA
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Declare volumes for user-defined paths
VOLUME ["/work/binary_data", "/work/models/roberta_pretrained", "/work/models/ramen_jur_roberta"]

# Set the command to be run on container start
CMD ["bash", "/work/train_roberta_ramen.sh"]
