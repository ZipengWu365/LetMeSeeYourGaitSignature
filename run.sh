#!/bin/bash
# =============================================================================
# CAPTURE-24 Gait Filter Pipeline - Linux One-Click Runner
# =============================================================================
# Usage:
#   chmod +x run.sh
#   ./run.sh                              # Full run (looks for ../capture24/prepared_data)
#   ./run.sh --data-dir /path/to/data     # Specify data directory
#   ./run.sh --quick-test                 # Quick test (10k samples)
#   ./run.sh --skip-minirocket            # Skip MiniRocket (saves memory)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PROJECT_ID="GF_LINUX_$(date +%Y%m%d_%H%M%S)"
QUICK_TEST=false
SKIP_MINIROCKET=false
INSTALL_DEPS=true
DATA_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            PROJECT_ID="GF_QUICKTEST"
            shift
            ;;
        --skip-minirocket)
            SKIP_MINIROCKET=true
            shift
            ;;
        --no-install)
            INSTALL_DEPS=false
            shift
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir PATH    Path to prepared_data directory (containing X.npy, Y.npy)"
            echo "  --quick-test       Run with 10k samples only (for testing)"
            echo "  --skip-minirocket  Skip MiniRocket features (saves ~40GB RAM)"
            echo "  --no-install       Skip dependency installation"
            echo "  --project-id ID    Set project ID for logs"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Example:"
            echo "  ./run.sh --data-dir ../capture24/prepared_data"
            echo "  ./run.sh --data-dir /home/user/data/capture24/prepared_data --quick-test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CAPTURE-24 Gait Filter Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "Quick Test: ${YELLOW}$QUICK_TEST${NC}"
echo -e "Skip MiniRocket: ${YELLOW}$SKIP_MINIROCKET${NC}"
echo ""

# Get script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}Script directory:${NC} $SCRIPT_DIR"

# Auto-detect data directory if not specified
if [ -z "$DATA_DIR" ]; then
    # Try common locations
    if [ -d "$SCRIPT_DIR/../../capture24/prepared_data" ]; then
        DATA_DIR="$SCRIPT_DIR/../../capture24/prepared_data"
    elif [ -d "$SCRIPT_DIR/../prepared_data" ]; then
        DATA_DIR="$SCRIPT_DIR/../prepared_data"
    elif [ -d "$SCRIPT_DIR/prepared_data" ]; then
        DATA_DIR="$SCRIPT_DIR/prepared_data"
    elif [ -d "./prepared_data" ]; then
        DATA_DIR="./prepared_data"
    else
        echo -e "${RED}Error: Could not find prepared_data directory!${NC}"
        echo ""
        echo "Please specify the path using --data-dir:"
        echo "  ./run.sh --data-dir /path/to/prepared_data"
        echo ""
        echo "The directory should contain: X.npy, Y.npy, P.npy"
        echo ""
        echo "To get the data, see: https://github.com/OxWearables/capture24"
        exit 1
    fi
fi

# Convert to absolute path
DATA_DIR="$( cd "$DATA_DIR" && pwd )"
echo -e "${BLUE}Data directory:${NC} $DATA_DIR"
echo ""

# Check Python
echo -e "${BLUE}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "  Found: ${GREEN}$PYTHON_VERSION${NC}"

# Install dependencies
if [ "$INSTALL_DEPS" = true ]; then
    echo ""
    echo -e "${BLUE}[2/5] Installing dependencies...${NC}"
    
    # Core dependencies
    $PYTHON -m pip install --quiet --upgrade pip
    $PYTHON -m pip install --quiet numpy scipy pandas scikit-learn joblib matplotlib statsmodels
    
    # Time-series packages
    $PYTHON -m pip install --quiet pyts xgboost
    
    # MrSQM (requires FFTW3 on Linux)
    echo -e "  Installing MrSQM..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y libfftw3-dev > /dev/null 2>&1 || true
    elif command -v yum &> /dev/null; then
        sudo yum install -y fftw-devel > /dev/null 2>&1 || true
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y fftw-devel > /dev/null 2>&1 || true
    fi
    $PYTHON -m pip install --quiet mrsqm || echo -e "${YELLOW}  Warning: MrSQM install failed, will skip MrSQM classifiers${NC}"
    
    # Optional: sktime for MiniRocket
    if [ "$SKIP_MINIROCKET" = false ]; then
        echo -e "  Installing sktime (for MiniRocket)..."
        $PYTHON -m pip install --quiet sktime || echo -e "${YELLOW}  Warning: sktime install failed, will skip MiniRocket${NC}"
    fi
    
    echo -e "  ${GREEN}Dependencies installed!${NC}"
else
    echo -e "${YELLOW}[2/5] Skipping dependency installation (--no-install)${NC}"
fi

# Check data files
echo ""
echo -e "${BLUE}[3/5] Checking data files...${NC}"

if [ ! -f "$DATA_DIR/X.npy" ]; then
    echo -e "${RED}Error: X.npy not found in $DATA_DIR${NC}"
    echo ""
    echo "Please download CAPTURE-24 data first:"
    echo "  git clone https://github.com/OxWearables/capture24.git"
    echo "  cd capture24 && python prepare_data.py"
    exit 1
fi

if [ ! -f "$DATA_DIR/Y.npy" ]; then
    echo -e "${RED}Error: Y.npy not found in $DATA_DIR${NC}"
    exit 1
fi

echo -e "  ${GREEN}Data files found!${NC}"
echo -e "  - X.npy: $(du -h $DATA_DIR/X.npy | cut -f1)"
echo -e "  - Y.npy: $(du -h $DATA_DIR/Y.npy | cut -f1)"

# Show system info
echo ""
echo -e "${BLUE}[4/5] System information...${NC}"
echo -e "  CPU cores: $(nproc)"
echo -e "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo -e "  Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Run pipeline
echo ""
echo -e "${BLUE}[5/5] Running pipeline...${NC}"
echo -e "${YELLOW}----------------------------------------${NC}"

cd "$SCRIPT_DIR"

# Build command
CMD="$PYTHON run_pipeline.py --project-id $PROJECT_ID --prepared-dir $DATA_DIR"

if [ "$QUICK_TEST" = true ]; then
    CMD="$CMD --quick-test"
fi

if [ "$SKIP_MINIROCKET" = true ]; then
    CMD="$CMD --skip-minirocket"
fi

echo -e "Command: ${GREEN}$CMD${NC}"
echo ""

# Execute
START_TIME=$(date +%s)

$CMD

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${YELLOW}----------------------------------------${NC}"
echo -e "${GREEN}Pipeline completed!${NC}"
echo -e "Total time: ${GREEN}${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "Output files:"
echo -e "  - Artifacts: ${BLUE}$SCRIPT_DIR/artifacts/gait_filter/${NC}"
echo -e "  - Logs: ${BLUE}$SCRIPT_DIR/logs/${NC}"
echo -e "  - Report: ${BLUE}$SCRIPT_DIR/${PROJECT_ID}_final_report.md${NC}"