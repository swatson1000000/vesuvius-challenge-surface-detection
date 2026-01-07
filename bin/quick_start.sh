#!/bin/bash
# Quick start script for Option C implementation

set -e

echo "========================================="
echo "Option C: nnU-Net + Topology-Aware Loss"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source /home/swatson/work/MachineLearning/.venv/bin/activate
fi

echo "Installing required dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || echo "PyTorch already installed"
pip install -q tifffile scikit-image scipy pyyaml tqdm connected-components-3d || echo "Some packages already installed"

echo ""
echo "✓ Dependencies installed"
echo ""

# Set data directory
DATA_DIR="/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection"

echo "=== Quick Tests ==="
echo ""

# Test topology losses
echo "1. Testing topology losses..."
python topology_losses.py 2>&1 | tail -n 5
echo "   ✓ Topology losses working"
echo ""

# Test model
echo "2. Testing model architecture..."
python nnunet_topo_wrapper.py 2>&1 | tail -n 5
echo "   ✓ Model architecture working"
echo ""

# Test post-processing
echo "3. Testing post-processing..."
python morphology_postprocess.py 2>&1 | tail -n 5
echo "   ✓ Post-processing working"
echo ""

echo "=== All tests passed! ==="
echo ""
echo "Next steps:"
echo "  1. Review the implementation in this directory"
echo "  2. Move to bin/ directory when ready"
echo "  3. Start training with:"
echo "     python train.py --config config.yaml --data_dir $DATA_DIR --fold 0"
echo ""
echo "  4. Run inference with:"
echo "     python inference.py --checkpoint checkpoints/fold_0/best_model.pth \\"
echo "                         --input $DATA_DIR/test_images \\"
echo "                         --output predictions/"
echo ""
