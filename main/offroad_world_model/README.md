# Off-Road Sudden Accident Video Generation by World Models

复现 ACCV 2026 Paper #41: **Video Generation of Sudden Off-Road Accidents by World Models for Autonomous Driving**

## 项目结构

```
offroad_world_model/
├── models/
│   ├── scene_encoder.py        # 多模态越野场景编码器 (Sec 3.2)
│   ├── world_dynamics.py       # 地形感知世界动力学 (Sec 3.2)
│   ├── accident_controller.py  # 条件事故控制器 (Sec 3.3)
│   ├── diffusion_decoder.py    # 光流引导扩散解码器 (Sec 3.3)
│   └── world_model.py          # 完整世界模型 (组合以上4个模块)
├── data/
│   └── demo_utils.py           # Demo数据生成工具
├── utils/
│   ├── optical_flow.py         # 光流计算工具
│   ├── metrics.py              # 评估指标 (PSNR, SSIM, Flow Error, Traj Score)
│   └── visualization.py        # 可视化工具
├── configs/
│   └── default.yaml            # 配置文件
├── inference.py                # 推理入口脚本
└── requirements.txt            # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Demo（合成场景）

```bash
# 突发侧滑事故
python inference.py --demo --accident_type sudden_skidding

# 动物入侵事故
python inference.py --demo --accident_type animal_intrusion

# 落石事故
python inference.py --demo --accident_type rockfall

# 地形塌陷事故
python inference.py --demo --accident_type terrain_collapse

# 使用真实图片作为条件帧
python inference.py --conditioning_image path/to/offroad.jpg --accident_type sudden_skidding
```

### 3. 查看所有事故类型

```bash
python inference.py --list_accidents
```

## 4种事故类型

| ID | 名称 | 描述 |
|----|------|------|
| 0 | animal_intrusion | 动物突然闯入行驶路径 |
| 1 | rockfall | 斜坡落石砸向车辆 |
| 2 | terrain_collapse | 车辆前方/下方地面塌陷 |
| 3 | sudden_skidding | 摩擦突变导致失控侧滑 |

## 模型架构

参考论文 Figure 2 的4个核心模块：

1. **SceneEncoder**: ResNet-18(预训练) + Temporal Transformer + Cross-Attention Fusion
2. **WorldDynamics**: Terrain Physical Embedding + Transformer预测未来潜状态
3. **AccidentController**: 事故控制向量注入 + 物理演化规则(摩擦突变模型)
4. **DiffusionDecoder**: 光流引导条件扩散生成 (预训练SVD backbone)

## 预训练模型使用情况

| 模块 | 预训练模型 | 来源 |
|------|-----------|------|
| ResNet-18 | ImageNet pretrained | torchvision |
| SVD Decoder | Stable Video Diffusion | stabilityai/svd-img2vid |
| RAFT (可选) | Optical Flow | torchvision |

## 训练

训练脚本将在数据集就绪后添加。目前仅支持推理模式。

当前使用预训练模型权重运行，暂不支持在自己的数据集上训练。
后续可添加训练功能（需要8×A100 80GB GPU集群）。

## 论文关键结果

| 指标 | 论文值 | 说明 |
|------|--------|------|
| PSNR | 32.74 dB | 生成质量 |
| SSIM | 0.961 | 结构相似性 |
| LPIPS | 0.038 | 感知距离 |
| 光流误差 | 0.049 | 时序一致性 |
| 轨迹合理性 | 0.937 | 物理合理性 |
| 感知mAP提升 | +14.6% | 下游安全验证 |
| 失控率降低 | -18.4% | 下游安全验证 |

## License

This implementation is for research purposes only.
