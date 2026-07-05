# Off-Road Sudden Accident Video Generation by World Models

ACCV 2026 Paper #41: **Video Generation of Sudden Off-Road Accidents by World Models for Autonomous Driving**

## project structure

```
offroad_world_model/
├── models/
│   ├── scene_encoder.py        # Multi-modal Off-road Scene Encoder (Sec 3.2)
│   ├── world_dynamics.py       # Terrain-sensing World Dynamics (Sec 3.2)
│   ├── accident_controller.py  # Terrain Condition Accident Controller (Sec 3.3)
│   ├── diffusion_decoder.py    # Flow-guided diffusion decoder (Sec 3.3)
│   └── world_model.py          # Complete World Model (by combining the above four modules)
├── data/
│   └── demo_utils.py           # Demo data generation tool
├── utils/
│   ├── optical_flow.py         # Flow calculation tool
│   ├── metrics.py              # evaluation index (PSNR, SSIM, Flow Error, Traj Score)
│   └── visualization.py        # visual tool
├── configs/
│   └── default.yaml            # configuration files
├── inference.py                # Inference entry script
└── requirements.txt            # dependency
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Demo (Synthetic Scenarios)

```bash
# Sudden skidding accident
python inference.py --demo --accident_type sudden_skidding
# Animal intrusion accident
python inference.py --demo --accident_type animal_intrusion
# Rockfall accident
python inference.py --demo --accident_type rockfall
# Terrain collapse accident
python inference.py --demo --accident_type terrain_collapse
# Use real images as conditional frames
python inference.py --conditioning_image path/to/offroad.jpg --accident_type sudden_skidding
```

### 3. List All Accident Categories

```bash
python inference.py --list_accidents
```

## Four Supported Accident Categories

| ID | Name | Description |
|----|------|------|
| 0 | animal_intrusion | Wild animals suddenly break into the driving path |
| 1 | rockfall | Falling rocks from slopes impact the vehicle |
| 2 | terrain_collapse | Ground collapse occurs in front of or underneath the vehicle |
| 3 | sudden_skidding | Vehicle loss of control and skidding caused by abrupt friction change |

## Model Architecture

Refer to Figure 2 in the original paper for the 4 core modules:

1. **SceneEncoder**: Pre-trained ResNet-18 + Temporal Transformer + Cross-Attention Fusion
2. **WorldDynamics**: Terrain Physical Embedding + Transformer for future latent state prediction
3. **AccidentController**: Accident control vector injection + physical evolution rules (abrupt friction model)
4. **DiffusionDecoder**: Optical-flow guided conditional video generation (pre-trained SVD backbone)

## Pre-trained Model Utilization

| Module | Pre-trained Model | Source |
|------|-----------|------|
| ResNet-18 | ImageNet pretrained | torchvision |
| SVD Decoder | Stable Video Diffusion | stabilityai/svd-img2vid |
| RAFT (Optional) | Optical Flow | torchvision |

## Training

requires an 8×A100 80GB GPU cluster

## Key Experimental Results from the Paper

| Metric | Paper Value | Explanation |
|------|--------|------|
| PSNR | 32.74 dB | Generation visual quality |
| SSIM | 0.961 | Structural similarity |
| LPIPS | 0.038 | Perceptual distance |
| Optical Flow Error | 0.049 | Temporal consistency |
| Trajectory Rationality Score | 0.937 | Physical plausibility of motion |
| Perception mAP Improvement | +14.6% | Downstream driving safety verification |
| Accident Out-of-Control Rate Reduction | -18.4% | Downstream driving safety verification |

## License

This implementation is for research purposes only.
