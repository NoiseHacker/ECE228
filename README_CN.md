# UCSD🔱 ECE228：机器学习在物理应用（2025 春季学期）

[English Version](README.md) | [中文版](README_CN.md)

欢迎来到 **ECE228：Machine Learning for Physical Applications** 课程仓库！  
本仓库收录了本学期全部作业与期末项目，内容涵盖稀疏回归、注意力机制、神经算子，以及面向真实世界的时空预测任务。

## 📐 作业一：稀疏线性回归
- 纯 NumPy 实现最小二乘与 **LASSO**  
- 编写 **ISTA**（迭代阈值-软阈值算法）求解 L1 正则化目标  
- 与 `scikit-learn` 结果对比  
- 在合成与真实数据集上完成实验和可视化  

## 🧠 作业二：Transformer 与注意力
- 以 PyTorch 手写 QKV 计算、**多头注意力**、位置编码等核心模块  
- 训练轻量级 Encoder 进行文本分类  
- 绘制注意力热图，分析多头冗余与模型泛化  

## 🌊 作业三：神经算子
- 实现 **Fourier Neural Operator (FNO)**，学习函数到函数的映射  
- 在 2-D Darcy 流体数据上验证跨分辨率泛化能力  
- 将 FNO 与 UNet 等基线比较，分析频域卷积优势  

## 🚦 期末项目：自适应图时空 Transformer 网络（**ASTTN-Weather**）
**目标：** 同时捕获道路拓扑与天气影响，实现高精度短时交通流量预测。  

| 模块 | 关键实现 |
| :-: | :- |
| **模型** | 基于 **ASTTN** [[论文](https://arxiv.org/abs/2207.05064)] 扩展：<br>• 采用 GIS 静态邻接矩阵<br>• 空间+时间联合自注意力<br>• 融合温度、降水、风速等多模态天气特征<br>• 轻量输出头 + 多步加权损失 |
| **变体** | Baseline ASTTN、ASTTN-Weather（本项目）及多种消融（是否使用天气/自适应边） |
| **数据集** | 1 min 采样的环形检测器流量/速度 + NOAA 每小时天气；对齐至 15 min |
| **结果** | 相比原版 ASTTN，ASTTN-Weather 在 15 min 预测上 **MAE↓9 %，R²↑6 %** |
| **复现** | `train.py` 兼容 CPU（PyTorch 2.7.1）；YAML 配置，自动保存日志/权重 |
| **可视化** | 支持绘制注意力热图及不同天气条件下边权变化 |

HAVE FUN! 🚀
