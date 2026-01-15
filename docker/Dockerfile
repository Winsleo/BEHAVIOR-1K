FROM nvcr.io/nvidia/isaac-sim:4.5.0

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
RUN rm -rf /isaac-sim/extscache/omni.kit.pip_archive*/pip_prebundle/websockets*
RUN /isaac-sim/python.sh -m pip install click~=8.1.3

# Default to /BEHAVIOR-1K/datasets (can be overridden via docker-compose environment variable)
ENV OMNIGIBSON_DATA_PATH=/BEHAVIOR-1K/datasets

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n behavior -c conda-forge python=3.10
RUN micromamba shell init --shell=bash

# Install numpy and setuptools first (required before other packages)
RUN micromamba run -n behavior pip install "numpy<2" "setuptools<=79"

# Install torch
RUN micromamba run -n behavior micromamba install \
  pytorch torchvision pytorch-cuda=11.8 \
  -c pytorch -c nvidia -c conda-forge

# Install curobo for primitives support. This can normally be installed when OmniGibson is pip
# installed, but we need to install it beforehand here so that it doesn't
# have to happen on every time a CI action is run (otherwise it's just
# very slow).
# This also allows us to uninstall the cuda toolkit after curobo is built
# to save space (meaning curobo will not be able to be rebuilt at runtime).
# Here we also compile this such that it is compatible with GPU architectures
# Turing, Ampere, and Ada; which correspond to 20, 30, and 40 series GPUs.
# We also suppress the output of the installation to avoid the log limit.
RUN wget --no-verbose -O /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \ 
  dpkg -i /cuda-keyring.deb && rm /cuda-keyring.deb && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-11-8 && \
  TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6+PTX' PATH=/usr/local/cuda-11.8/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    micromamba run -n behavior pip install \
    git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73#egg=nvidia_curobo \
    --no-build-isolation > /dev/null && \
  apt-get remove -y cuda-toolkit-11-8 && apt-get autoremove -y && apt-get autoclean -y && rm -rf /var/lib/apt/lists/*

# Make sure isaac gets properly sourced every time behavior gets called
ARG CONDA_ACT_FILE="/micromamba/envs/behavior/etc/conda/activate.d/env_vars.sh"
RUN mkdir -p "/micromamba/envs/behavior/etc/conda/activate.d"
RUN touch $CONDA_ACT_FILE

# Use bash instead of sh to ensure proper sourcing
RUN echo '#!/bin/bash' > $CONDA_ACT_FILE
RUN echo "source /isaac-sim/setup_conda_env.sh" >> $CONDA_ACT_FILE
RUN chmod +x $CONDA_ACT_FILE

# Add to bashrc: activate conda environment first, then source Isaac Sim setup
# Note: setup_conda_env.sh removes Kit Python from PYTHONPATH to avoid conflicts with conda
RUN echo "micromamba activate behavior" >> /root/.bashrc
RUN echo "source /isaac-sim/setup_conda_env.sh" >> /root/.bashrc

# Copy over BEHAVIOR-1K source
ADD . /BEHAVIOR-1K
WORKDIR /BEHAVIOR-1K

# Set the shell
SHELL ["micromamba", "run", "-n", "behavior", "/bin/bash", "--login", "-c"]

# Set NVIDIA EULA acceptance for Isaac Sim installation
ENV OMNI_KIT_ACCEPT_EULA=YES

# Optionally install components (e.g. unless the DEV_MODE flag is set) or
# remove the source code if we are in dev mode and change the workdir
ARG DEV_MODE
ENV DEV_MODE=${DEV_MODE}
ARG INSTALL_EXTRAS=eval,primitives
ENV INSTALL_EXTRAS=${INSTALL_EXTRAS}
ARG JOYLO_INSTALL=false
ENV JOYLO_INSTALL=${JOYLO_INSTALL}
ARG WORKDIR_PATH=/BEHAVIOR-1K

RUN if [ "$DEV_MODE" != "1" ]; then \
      echo "Installing BDDL..."; \
      cd /BEHAVIOR-1K/bddl3 && \
      micromamba run -n behavior pip install -e .; \
      echo "Installing OmniGibson with extras: $INSTALL_EXTRAS"; \
      cd /BEHAVIOR-1K/OmniGibson && \
      micromamba run -n behavior pip install -e ".[$INSTALL_EXTRAS]"; \
      echo 'echo "To download datasets, run: OMNIGIBSON_NO_OMNIVERSE=1 python -m omnigibson.utils.asset_utils --download_behavior_1k_assets"' >> /root/.bashrc; \
      if [ "$JOYLO_INSTALL" = "true" ]; then \
        echo "Installing JoyLo..."; \
        cd /BEHAVIOR-1K/joylo && \
        if [ -f "requirements.txt" ]; then \
          micromamba run -n behavior pip install -r requirements.txt; \
        fi && \
        micromamba run -n behavior pip install -e .; \
      fi; \
    else \
      WORKDIR_PATH=/; \
      cd / && rm -rf /BEHAVIOR-1K; \
    fi

# Reset the WORKDIR based on whether or not we are in dev mode
WORKDIR ${WORKDIR_PATH}

ENTRYPOINT ["micromamba", "run", "-n", "behavior"]
CMD ["/bin/bash"]
