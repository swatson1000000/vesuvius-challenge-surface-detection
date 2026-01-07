# Development Setup

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd vesuvius-challenge-surface-detection
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Data
Download competition data from Kaggle:
```bash
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip
```

## Directory Structure

```
vesuvius-challenge-surface-detection/
├── .git/                         # Git repository
├── .venv/                        # Virtual environment (gitignored)
├── bin/                          # Main implementation
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference pipeline
│   ├── nnunet_topo_wrapper.py    # Model wrapper
│   ├── topology_losses.py        # Custom losses
│   ├── morphology_postprocess.py # Post-processing
│   ├── config.yaml               # Configuration
│   ├── checkpoints/              # Model checkpoints (gitignored)
│   └── README.md                 # Implementation docs
├── train_images/                 # Training CT scans (gitignored)
├── train_labels/                 # Training masks (gitignored)
├── test_images/                  # Test data (gitignored)
├── option_c_implementation/      # Alternative approaches
├── topological-metrics-kaggle/   # Competition metrics
├── log/                          # Training logs (gitignored)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── README.md                     # Project overview
├── COMPETITION_INFO.md           # Competition details
└── PLAN.md                       # Strategy document
```

## Git Workflow

### Initial Setup
```bash
git init
~/.claude/install-hooks.sh  # Install Claude code review hooks
```

### Making Changes
```bash
git add .
git commit -m "Descriptive message"
git push origin main
```

The pre-push hook will automatically review code and block pushes with critical/high severity issues.

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and modular

### Testing
```bash
pytest tests/
```

### Logging
Use Python's logging module:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Training

### Quick Start
```bash
cd bin
bash quick_start.sh
```

### Full Training
```bash
cd bin
python train.py --config config.yaml --fold 0
```

### Configuration
Edit `bin/config.yaml` to adjust:
- Model architecture
- Loss function weights
- Training hyperparameters
- Data augmentation
- Post-processing parameters

## Inference

### Run Inference
```bash
cd bin
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input ../test_images/ \
    --output ../predictions/
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir log/
```

### Weights & Biases
Configure in `config.yaml`:
```yaml
wandb:
  project: vesuvius-challenge
  entity: your-username
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Use gradient checkpointing
- Enable mixed precision training

### Slow Training
- Use SSD for data storage
- Increase number of workers in DataLoader
- Enable data caching

### Topology Errors
- Adjust post-processing thresholds
- Check label quality
- Verify metric calculations

## Resources

- [Competition Page](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
- [PLAN.md](PLAN.md) - Comprehensive strategy
- [COMPETITION_INFO.md](COMPETITION_INFO.md) - Official details
- [bin/README.md](bin/README.md) - Implementation details

## Support

For issues or questions:
1. Check documentation files
2. Review competition discussion forum
3. Open a GitHub issue
