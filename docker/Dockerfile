FROM nvcr.io/nvidia/isaac-sim:4.5.0

# CUDA version configuration (must match PyTorch CUDA version)
# Note: Using CUDA 12.8 for Blackwell (RTX 50 series) GPU support
ARG CUDA_VERSION=12.8
ARG CUDA_PKG_VERSION=12-8
# Set up all the prerequisites.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl git wget \
    g++ cmake pkg-config libeigen3-dev wget libyaml-cpp-dev castxml pypy3 \
    libhidapi-dev android-tools-adb libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym*
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch*
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/functorch*
RUN rm -rf /isaac-sim/kit/extscore/omni.kit.pip_archive/pip_prebundle/numpy*
RUN /isaac-sim/python.sh -m pip install click~=8.1.3

# Default to /BEHAVIOR-1K/datasets (can be overridden via docker-compose environment variable)
ENV OMNIGIBSON_DATA_PATH=/BEHAVIOR-1K/datasets

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n behavior -c conda-forge python=3.10 && micromamba shell init --shell=bash

# Install torch with specific CUDA version via pip (most reliable method)
# Convert CUDA version to short format: 11.8 -> 118
RUN CUDA_VER_SHORT=$(echo ${CUDA_VERSION} | sed 's/\.//g') && \
    micromamba run -n behavior pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}

# Install curobo for primitives support. This can normally be installed when OmniGibson is pip
# installed, but we need to install it beforehand here so that it doesn't
# have to happen on every time a CI action is run (otherwise it's just
# very slow).
# This also allows us to uninstall the cuda toolkit after curobo is built
# to save space (meaning curobo will not be able to be rebuilt at runtime).
# Here we also compile this such that it is compatible with GPU architectures:
#   - 7.0: V100 (Volta)
#   - 7.5: RTX 20 series (Turing)
#   - 8.0: A100, RTX 30 series (Ampere)
#   - 8.6: RTX 30 series variants (Ampere)
#   - 8.9: RTX 40 series (Ada Lovelace)
#   - 10.0: RTX 50 series (Blackwell)
# We also suppress the output of the installation to avoid the log limit.
RUN wget --no-verbose -O /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \ 
    dpkg -i /cuda-keyring.deb && rm /cuda-keyring.deb && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-${CUDA_PKG_VERSION} && \
    TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;10.0+PTX' PATH=${CUDA_HOME}/bin:$PATH LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH \
    micromamba run -n behavior pip install \
    git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73#egg=nvidia_curobo \
    --no-build-isolation > /dev/null && \
    apt-get remove -y cuda-toolkit-${CUDA_PKG_VERSION} && apt-get autoremove -y && apt-get autoclean -y && rm -rf /var/lib/apt/lists/*

# Make sure isaac gets properly sourced every time behavior gets called
ARG CONDA_ACT_FILE="/micromamba/envs/behavior/etc/conda/activate.d/env_vars.sh"
RUN mkdir -p "/micromamba/envs/behavior/etc/conda/activate.d" &&touch $CONDA_ACT_FILE

# Use bash instead of sh to ensure proper sourcing
RUN echo '#!/bin/bash' > $CONDA_ACT_FILE && \
    echo "source /isaac-sim/setup_conda_env.sh" >> $CONDA_ACT_FILE && \
    echo "micromamba activate behavior" >> /root/.bashrc

# Copy over BEHAVIOR-1K source
ADD . /BEHAVIOR-1K
WORKDIR /BEHAVIOR-1K

# Set the shell
SHELL ["micromamba", "run", "-n", "behavior", "/bin/bash", "--login", "-c"]

# Set NVIDIA EULA acceptance for Isaac Sim installation
ENV OMNI_KIT_ACCEPT_EULA=YES

ARG INSTALL_EXTRAS=eval,primitives
ENV INSTALL_EXTRAS=${INSTALL_EXTRAS}
ARG JOYLO_INSTALL=false
ENV JOYLO_INSTALL=${JOYLO_INSTALL}
ARG WORKDIR_PATH=/BEHAVIOR-1K

RUN echo "Installing BDDL..."; \
    micromamba run -n behavior pip install -e /BEHAVIOR-1K/bddl3; \
    if [ "$JOYLO_INSTALL" = "true" ]; then \
        echo "Installing JoyLo..."; \
        micromamba run -n behavior pip install -e /BEHAVIOR-1K/joylo; \
    fi; \
    echo "Installing OmniGibson with extras: $INSTALL_EXTRAS"; \
    micromamba run -n behavior pip install -e "/BEHAVIOR-1K/OmniGibson[$INSTALL_EXTRAS]"; \
    echo 'echo "To download datasets, run: OMNIGIBSON_NO_OMNIVERSE=1 python -m omnigibson.utils.asset_utils --download_behavior_1k_assets"' >> /root/.bashrc;

# Reset the WORKDIR based on whether or not we are in dev mode
WORKDIR ${WORKDIR_PATH}

ENTRYPOINT ["micromamba", "run", "-n", "behavior"]
CMD ["/bin/bash"]
