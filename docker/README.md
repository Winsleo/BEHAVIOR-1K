# BEHAVIOR-1K Docker Image

This Dockerfile creates a containerized environment for BEHAVIOR-1K with all necessary dependencies including OmniGibson, BDDL, and optional JoyLo support.

## Base Image

Based on NVIDIA Isaac Sim 4.5.0 (`nvcr.io/nvidia/isaac-sim:4.5.0`)

## Build Arguments

- `DEV_MODE`: Set to `1` to enable development mode (source code will be removed from container, useful for CI/CD)
- `INSTALL_EXTRAS`: Comma-separated list of OmniGibson extras to install (default: `eval,primitives`)
  - Available extras: `dev`, `primitives`, `eval`
- `JOYLO_INSTALL`: Set to `true` to install JoyLo teleoperation interface (default: `false`)

## Building the Image

### Basic build (with eval and primitives support):
```bash
docker build -t behavior-1k:latest -f docker/Dockerfile .
```

### Build with JoyLo support:
```bash
docker build -t behavior-1k:latest \
  --build-arg JOYLO_INSTALL=true \
  -f docker/Dockerfile .
```

### Build in development mode (for CI/CD):
```bash
docker build -t behavior-1k:latest \
  --build-arg DEV_MODE=1 \
  -f docker/Dockerfile .
```

### Build with custom extras:
```bash
docker build -t behavior-1k:latest \
  --build-arg INSTALL_EXTRAS="eval,primitives,dev" \
  -f docker/Dockerfile .
```

## Running the Container

### Using run_docker.sh Script (Easiest)

The `run_docker.sh` script provides the easiest way to run BEHAVIOR-1K containers with automatic setup:

#### Basic usage:
```bash
cd docker
./run_docker.sh
```

#### Headless mode (for evaluation):
```bash
./run_docker.sh --headless
```

#### Development mode:
```bash
./run_docker.sh --dev
```

#### With JoyLo support:
```bash
./run_docker.sh --joylo
```

#### Custom data directory:
```bash
./run_docker.sh --data-dir /path/to/custom/data
```

#### Force rebuild image:
```bash
./run_docker.sh --build
./run_docker.sh --build --joylo  # Rebuild and start with JoyLo
```

#### Stop containers:
```bash
./run_docker.sh --stop
```

#### Clean up (stop and remove containers, networks, volumes):
```bash
./run_docker.sh --clean  # Will prompt for confirmation
```

#### Show help:
```bash
./run_docker.sh --help
```

The script automatically:
- Automatically accepts NVIDIA EULA (no user confirmation required)
- Creates necessary data directory structure
- Sets up X11 forwarding for GUI mode
- Builds the image automatically if it doesn't exist (use `--build` to force rebuild)
- Manages cache directories for better performance

**Note**: The script does not rebuild the image every time. It only builds if the image doesn't exist. Use the `--build` flag to force a rebuild when you've updated the Dockerfile or source code.

### Using Docker Compose (Recommended)

Docker Compose provides an easier way to manage containers with pre-configured settings.

#### Basic usage:
```bash
cd docker
docker compose up -d behavior-1k
docker compose exec behavior-1k /bin/bash
```

#### Development mode:
```bash
docker compose --profile dev up -d behavior-1k-dev
docker compose exec behavior-1k-dev /bin/bash
```

#### With JoyLo support:
```bash
docker compose --profile joylo up -d behavior-1k-joylo
docker compose exec behavior-1k-joylo /bin/bash
```

#### Headless evaluation mode:
```bash
docker compose --profile eval up -d behavior-1k-eval
docker compose exec behavior-1k-eval /bin/bash
```

#### Stop containers:
```bash
docker compose down
```

### Using Docker directly

#### Basic run:
```bash
docker run --gpus all -it --rm behavior-1k:latest
```

#### Run with data volume mounted:
```bash
docker run --gpus all -it --rm \
  -v /path/to/data:/data \
  behavior-1k:latest
```

#### Run with X11 forwarding (for GUI):
```bash
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  behavior-1k:latest
```

#### Run with interactive shell and mounted workspace:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  behavior-1k:latest /bin/bash
```

## Environment Variables

- `OMNIGIBSON_DATA_PATH`: Path to data directory (default: `/data`)
- `OMNI_KIT_ACCEPT_EULA`: Automatically set to `YES` to accept NVIDIA EULA

## Installed Components

- **Python 3.10** via micromamba
- **PyTorch** with CUDA 11.8 support
- **BDDL** (Behavior Domain Definition Language)
- **OmniGibson** with eval and primitives support
- **Curobo** (for motion planning primitives)
- **JoyLo** (optional, if `JOYLO_INSTALL=true`)

## Docker Compose Services

The `docker-compose.yml` file provides several pre-configured services:

- **behavior-1k**: Default service with full BEHAVIOR-1K installation (default)
- **behavior-1k-dev**: Development mode service (use `--profile dev`)
- **behavior-1k-joylo**: Service with JoyLo teleoperation support (use `--profile joylo`)
- **behavior-1k-eval**: Headless evaluation service (use `--profile eval`)

### Docker Compose Configuration

The compose file automatically handles:
- GPU access via `deploy.resources.reservations.devices` (modern Docker Compose V2 approach)
- X11 forwarding for GUI applications
- Volume mounts for data and source code
- Environment variables configuration

**Note**: This configuration uses Docker Compose V2 (no `version` field required). GPU access is configured via device reservations rather than the deprecated `runtime: nvidia` option.

### Quick Start with Docker Compose

1. **Navigate to docker directory:**
   ```bash
   cd docker
   ```

2. **Start the default service (image will be built automatically if it doesn't exist):**
   ```bash
   docker compose up -d behavior-1k
   ```

   Or explicitly build first:
   ```bash
   docker compose build behavior-1k
   docker compose up -d behavior-1k
   ```

3. **Access the container:**
   ```bash
   docker compose exec behavior-1k /bin/bash
   ```

4. **View logs:**
   ```bash
   docker compose logs -f behavior-1k
   ```

5. **Stop the service:**
   ```bash
   docker compose down
   ```
   
   Or use the script:
   ```bash
   ./run_docker.sh --stop
   ```

6. **Clean up (stop and remove containers, networks, volumes):**
   ```bash
   docker compose down -v --remove-orphans
   ```
   
   Or use the script (with confirmation prompt):
   ```bash
   ./run_docker.sh --clean
   ```

### Service-Specific Usage

#### Development Mode
For development with source code mounted at runtime (container image doesn't include source code):
```bash
# Image will be built automatically if it doesn't exist
docker compose --profile dev up -d behavior-1k-dev
docker compose exec behavior-1k-dev /bin/bash

# Or explicitly build first:
docker compose --profile dev build behavior-1k-dev
docker compose --profile dev up -d behavior-1k-dev
docker compose exec behavior-1k-dev /bin/bash
```

#### JoyLo Teleoperation
For teleoperation with JoyCon support:
```bash
# Image will be built automatically if it doesn't exist
docker compose --profile joylo up -d behavior-1k-joylo
docker compose exec behavior-1k-joylo /bin/bash

# Or explicitly build first:
docker compose --profile joylo build behavior-1k-joylo
docker compose --profile joylo up -d behavior-1k-joylo
docker compose exec behavior-1k-joylo /bin/bash
```

#### Headless Evaluation
For running evaluations without GUI:
```bash
# Image will be built automatically if it doesn't exist
docker compose --profile eval up -d behavior-1k-eval
docker compose exec behavior-1k-eval /bin/bash

# Or explicitly build first:
docker compose --profile eval build behavior-1k-eval
docker compose --profile eval up -d behavior-1k-eval
docker compose exec behavior-1k-eval /bin/bash
```

### Customizing Docker Compose

You can override settings using environment variables:

```bash
# Override display for X11 forwarding
export DISPLAY=:0
docker compose up -d

# Use custom data directory (all volumes will use this path)
export DATA_PATH=/custom/path/to/data
docker compose up -d

# Combine multiple settings
export DATA_PATH=/custom/data
export DISPLAY=:0
docker compose up -d
```

The `DATA_PATH` environment variable controls the base path for all data directories. When set, all volume mounts will use this path instead of the default `./data`.

## Data Directory Structure and Environment Variables

The docker compose configuration uses the `DATA_PATH` environment variable (default: `./data`) to specify where all persistent data should be stored. You can customize this by setting the environment variable:

```bash
export DATA_PATH=/path/to/your/data
docker compose up -d
```

### Directory Structure

**Important**: The OmniGibson data root is set to `/BEHAVIOR-1K/datasets` via the `OMNIGIBSON_DATA_PATH` environment variable. Since the entire project root is mounted at `/BEHAVIOR-1K`, this directly uses the `BEHAVIOR-1K/datasets` directory. Cache directories are stored under `$DATA_PATH` (default: `./data`).

```
BEHAVIOR-1K/
├── datasets/              # ← OmniGibson data root (mapped to /data in container)
│   ├── behavior-1k-assets/
│   ├── 2025-challenge-task-instances/
│   └── omnigibson.key
│
docker/
└── data/                  # ← Cache directories (controlled by DATA_PATH)
    ├── outputs/           # Evaluation outputs, logs, and results
    └── isaac-sim/
        ├── cache/
        │   ├── kit/       # OmniKit cache (scene loading, asset processing)
        │   ├── ov/        # Omniverse cache (USD files, materials)
        │   ├── pip/       # Python package cache (speeds up pip installs)
        │   ├── glcache/   # NVIDIA OpenGL shader cache (GPU rendering)
        │   └── computecache/ # NVIDIA compute cache (CUDA kernels)
        ├── logs/          # Isaac Sim application logs and error reports
        ├── config/        # Isaac Sim user preferences and settings
        ├── data/          # Omniverse application data and extensions
        └── documents/     # User documents and saved projects
```

### Detailed Directory Descriptions

#### 1. `BEHAVIOR-1K/datasets/` → `/BEHAVIOR-1K/datasets` (Container Path via `OMNIGIBSON_DATA_PATH`)
**Purpose**: This is the **root data directory** for OmniGibson and BEHAVIOR-1K. It stores all datasets, assets, and task-related files.

**Note**: 
- This directory is located at `BEHAVIOR-1K/datasets/` in the project root
- It is accessed via `OMNIGIBSON_DATA_PATH=/BEHAVIOR-1K/datasets` environment variable
- No separate volume mapping needed - uses the project root mount (`../:/BEHAVIOR-1K:rw`)
- This simplifies configuration and makes the data location more intuitive

**What OmniGibson stores here:**
- **BEHAVIOR-1K Assets** (`behavior-1k-assets/`):
  - Scene models (USD files for houses, rooms, etc.)
  - Object models (3D models for all objects used in tasks)
  - Scene configurations and metadata
  - Object categories and specifications
  
- **Task Instances** (`2025-challenge-task-instances/`):
  - Pre-sampled task configurations
  - Task metadata (episodes.jsonl, test_instances.csv)
  - Scene-specific task instance files
  - Human demonstration statistics

- **Decryption Key** (`omnigibson.key`):
  - Required for accessing encrypted BEHAVIOR dataset assets

**How OmniGibson uses it:**
- `gm.DATA_PATH` (set via `OMNIGIBSON_DATA_PATH` environment variable)
- Used by `get_dataset_path()` to locate datasets
- Used by `get_task_instance_path()` to find task configurations
- Used by `get_scene_path()` to locate scene files
- Used by `get_category_path()` to find object models

**Example paths OmniGibson expects:**
```
/data/
├── behavior-1k-assets/
│   ├── scenes/              # Scene models (Rs_int, house_single_floor, etc.)
│   ├── objects/             # Object 3D models by category
│   └── metadata/            # Categories, specifications, etc.
├── 2025-challenge-task-instances/
│   ├── scenes/              # Task instance files per scene
│   └── metadata/            # Task metadata files
└── omnigibson.key           # Dataset decryption key
```

**Size**: Can be very large (50+ GBs) - contains all 3D assets
**Persistence**: **CRITICAL** - contains all simulation assets required for tasks
**Backup**: **Highly recommended** - losing this directory means re-downloading everything
**Download**: Use `python omnigibson/download_datasets.py` to populate this directory

#### 2. `outputs/` → `/BEHAVIOR-1K/outputs` (via project root mount)
**Purpose**: Stores evaluation results, training outputs, and generated data
- **Contents**:
  - Evaluation metrics and logs
  - Trained model checkpoints
  - Generated videos and visualizations
  - Experiment results
- **Location**: `BEHAVIOR-1K/outputs/` in project root (not in `docker/data/`)
- **Note**: No separate volume mapping needed - uses the project root mount (`../:/BEHAVIOR-1K:rw`)
- **Size**: Grows with usage
- **Persistence**: Important for preserving results

#### 3. `isaac-sim/cache/kit/` → `/isaac-sim/kit/cache/Kit`
**Purpose**: OmniKit runtime cache for scene loading and asset processing
- **Contents**: 
  - Processed scene files
  - Compiled asset metadata
  - Scene graph optimizations
- **Size**: Several GBs
- **Performance Impact**: High - significantly speeds up scene loading
- **Can Delete**: Yes, will be regenerated (slower first run)

#### 4. `isaac-sim/cache/ov/` → `/root/.cache/ov`
**Purpose**: Omniverse application cache
- **Contents**:
  - USD file caches
  - Material and texture caches
  - Extension metadata
- **Size**: Hundreds of MBs to GBs
- **Performance Impact**: Medium - improves asset loading
- **Can Delete**: Yes, will be regenerated

#### 5. `isaac-sim/cache/pip/` → `/root/.cache/pip`
**Purpose**: Python package manager cache
- **Contents**:
  - Downloaded Python wheel files
  - Package installation artifacts
- **Size**: Can be large (GBs) if many packages installed
- **Performance Impact**: High - speeds up pip installs significantly
- **Can Delete**: Yes, packages will re-download on next install

#### 6. `isaac-sim/cache/glcache/` → `/root/.cache/nvidia/GLCache`
**Purpose**: NVIDIA OpenGL shader cache
- **Contents**:
  - Compiled OpenGL shaders
  - GPU rendering optimizations
- **Size**: Hundreds of MBs
- **Performance Impact**: High - eliminates shader compilation delays
- **Can Delete**: Yes, shaders will recompile (first frame may be slow)

#### 7. `isaac-sim/cache/computecache/` → `/root/.nv/ComputeCache`
**Purpose**: NVIDIA CUDA compute kernel cache
- **Contents**:
  - Compiled CUDA kernels
  - GPU compute optimizations
- **Size**: Hundreds of MBs
- **Performance Impact**: High - speeds up physics and rendering
- **Can Delete**: Yes, kernels will recompile

#### 8. `isaac-sim/logs/` → `/root/.nvidia-omniverse/logs`
**Purpose**: Isaac Sim application logs
- **Contents**:
  - Error logs
  - Debug information
  - Crash reports
- **Size**: MBs to GBs (grows over time)
- **Performance Impact**: None
- **Can Delete**: Yes, for cleanup (logs will regenerate)

#### 9. `isaac-sim/config/` → `/root/.nvidia-omniverse/config`
**Purpose**: Isaac Sim user configuration and preferences
- **Contents**:
  - User preferences
  - Window layouts
  - Keyboard shortcuts
  - Application settings
- **Size**: Small (MBs)
- **Performance Impact**: None (but preserves your settings)
- **Can Delete**: Yes, but you'll lose custom settings

#### 10. `isaac-sim/data/` → `/root/.local/share/ov/data`
**Purpose**: Omniverse application data and extensions
- **Contents**:
  - Installed extensions
  - Application data files
  - Extension caches
- **Size**: Can be large (GBs) if many extensions installed
- **Performance Impact**: Low
- **Can Delete**: Yes, but extensions may need reinstallation

#### 11. `isaac-sim/documents/` → `/root/Documents`
**Purpose**: User documents and saved projects
- **Contents**:
  - Saved scenes
  - User projects
  - Custom configurations
- **Size**: Depends on usage
- **Performance Impact**: None
- **Can Delete**: Yes, but you'll lose saved work

### Using Custom Data Path

**Note**: The OmniGibson data root is **always** mapped to `BEHAVIOR-1K/datasets/`. The `DATA_PATH` environment variable only controls cache directories location.

To use a custom path for cache directories:

```bash
# Method 1: Environment variable
export DATA_PATH=/custom/path/to/cache
docker compose up -d

# Method 2: Inline with command
DATA_PATH=/custom/path/to/cache docker compose up -d

# Method 3: Create .env file in docker/ directory
echo "DATA_PATH=/custom/path/to/cache" > docker/.env
docker compose up -d
```

**Important Notes**:
- **OmniGibson data root** is set to `/BEHAVIOR-1K/datasets` via `OMNIGIBSON_DATA_PATH` environment variable
- No separate `/data` volume mapping needed - uses the project root mount directly
- `DATA_PATH` only controls where **cache directories** are stored (default: `docker/data/`)
- The datasets directory must be on a **local filesystem** (not AFS/NFS)
- Ensure sufficient disk space (datasets can be 50+ GB)
- Cache directories improve performance but can be deleted if needed
- Consider backing up the `BEHAVIOR-1K/datasets/` directory regularly

## Stopping and Cleaning Up Containers

### Stop Containers

Stop running containers without removing them:

```bash
# Using docker compose
docker compose down

# Using the script
./run_docker.sh --stop
```

### Clean Up Containers

Remove containers, networks, and volumes (with confirmation prompt):

```bash
# Using docker compose (no confirmation)
docker compose down -v --remove-orphans

# Using the script (with confirmation)
./run_docker.sh --clean
```

**Note**: The `--clean` option will:
- Stop all running containers
- Remove containers, networks, and volumes
- Remove orphaned containers
- **WARNING**: This will remove volumes, but your data in `BEHAVIOR-1K/datasets/` and cache directories are safe (they're mounted from host)

### Clean Up Docker Images

To remove unused Docker images and free up disk space:

```bash
# Remove unused images
docker image prune -a

# Remove all unused resources (containers, networks, images, build cache)
docker system prune -a --volumes
```

**Warning**: `docker system prune -a` will remove all unused images, including potentially useful ones. Use with caution.

## Notes

- The container uses micromamba for environment management
- Isaac Sim environment is automatically sourced when activating the conda environment
- CUDA toolkit is installed temporarily to build curobo, then removed to save space
- In development mode (`DEV_MODE=1`), source code is removed from the container image but mounted at runtime
- Docker Compose V2 automatically builds images if they don't exist when running `docker compose up`
- Use `--build` flag with `run_docker.sh` or `docker compose build` to force rebuild after Dockerfile changes
- Docker Compose automatically creates a `data` directory structure in the `docker/` folder for persistent storage
- The `run_docker.sh` script automatically manages X11 permissions and data directory setup
- Cache directories are mounted to improve performance on subsequent runs

## Troubleshooting

### GPU not detected:
Make sure to use `--gpus all` flag when running the container.

### Out of disk space:
The curobo build process requires significant disk space. Consider cleaning up unused Docker images:
```bash
docker system prune -a
```

### Permission issues:
If you encounter permission issues with mounted volumes, ensure proper permissions:
```bash
docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v /path/to/data:/data \
  behavior-1k:latest
```
