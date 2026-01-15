#!/usr/bin/env bash
set -e -o pipefail

BYellow='\033[1;33m'
BGreen='\033[1;32m'
Color_Off='\033[0m'

# Parse the command line arguments.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Note: OmniGibson data root is mapped to ../datasets (BEHAVIOR-1K/datasets)
# Cache directories still use DATA_PATH (default: ./data)
DEFAULT_DATA_DIR="$SCRIPT_DIR/data"
DATA_PATH=$DEFAULT_DATA_DIR
GUI=true
SERVICE="behavior-1k"
PROFILE=""
BUILD=false
STOP=false
CLEAN=false
HEADLESS=0

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--headless)
        GUI=false
        HEADLESS=1
        shift
        ;;
        --joylo)
        SERVICE="behavior-1k-joylo"
        PROFILE="--profile joylo"
        shift
        ;;
        --data-dir)
        DATA_PATH="$2"
        shift 2
        ;;
        --build)
        BUILD=true
        shift
        ;;
        --stop)
        STOP=true
        shift
        ;;
        --clean)
        CLEAN=true
        shift
        ;;
        --help)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -h, --headless          Run in headless mode (set OMNIGIBSON_HEADLESS=1)"
        echo "  --joylo                 Run with JoyLo support"
        echo "  --data-dir PATH         Specify custom data directory (default: ./data)"
        echo "  --build                 Force rebuild the Docker image"
        echo "  --stop                  Stop running containers"
        echo "  --clean                 Stop and remove containers, networks, volumes"
        echo "  --help                  Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                      # Run with GUI"
        echo "  $0 --headless           # Run headless (evaluation)"
        echo "  $0 --joylo              # Run with JoyLo teleoperation"
        echo "  $0 --data-dir /path     # Use custom cache directory"
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
done

echo -e "${BYellow}Data root: BEHAVIOR-1K/datasets"
echo -e "Cache: ${DATA_PATH}/isaac-sim/"
echo -e "Mode: $([ $HEADLESS -eq 1 ] && echo 'Headless' || echo 'GUI')${Color_Off}"
echo ""

# Create cache directory structure
mkdir -p "$DATA_PATH"/isaac-sim/{cache/{kit,ov,pip,glcache,computecache},logs,config,data,documents}

# Set up X11 forwarding for GUI mode
if [ "$GUI" = true ] ; then
    echo -e "${BGreen}Setting up X11 forwarding...${Color_Off}"
    xhost +local: 2>/dev/null || echo "Warning: Could not set xhost permissions."
fi

cd "$SCRIPT_DIR"

# Export environment variables for docker compose
export DATA_PATH
export OMNIGIBSON_HEADLESS=$HEADLESS

# Handle stop/clean operations
if [ "$STOP" = true ] || [ "$CLEAN" = true ]; then
    if [ "$STOP" = true ]; then
        echo -e "${BGreen}Stopping containers...${Color_Off}"
        docker compose $PROFILE stop $SERVICE 2>/dev/null || docker compose $PROFILE down
    fi
    if [ "$CLEAN" = true ]; then
        echo -e "${BYellow}WARNING: This will remove containers, networks, and volumes.${Color_Off}"
        read -p "Continue? [y/N] " yn
        case $yn in
            [Yy]* )
                docker compose $PROFILE down -v --remove-orphans
                echo -e "${BGreen}Cleanup done.${Color_Off}"
                ;;
            * ) exit 0 ;;
        esac
    fi
    exit 0
fi

# Build if requested
if [ "$BUILD" = true ]; then
    echo -e "${BGreen}Building image...${Color_Off}"
    docker compose $PROFILE build $SERVICE
fi

# Start container
echo -e "${BGreen}Starting container...${Color_Off}"
docker compose $PROFILE up -d $SERVICE
sleep 2

# Enter container
echo -e "${BGreen}Entering container...${Color_Off}"
docker compose $PROFILE exec $SERVICE /bin/bash

# Cleanup X11
if [ "$GUI" = true ] ; then
    xhost -local: 2>/dev/null || true
fi
