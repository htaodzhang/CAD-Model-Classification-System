
# 3D CAD Classification System / 3D CAD分类系统

This is a professional 3D CAD model classification platform built with PyQt5 and PythonOCC. It integrates advanced deep learning technologies, specifically UV-Net, to classify STEP format CAD models with high accuracy.

这是一个基于 PyQt5 和 PythonOCC 开发的专业3D CAD模型分类平台，集成了先进的深度学习技术（UV-Net），可对STEP格式的CAD模型进行高精度分类。

![System Screenshot](https://github.com/BrepMaster/CAD-Model-Classification-System/raw/main/1.png)

📦 **Download (Windows EXE version)**
链接: https://pan.baidu.com/s/19nqyOoqF7ECdOlHPnMVC9g?pwd=b9ax 
提取码: b9ax
**温馨提示**
如果本项目对您有所帮助，欢迎点击右上角 ⭐Star 支持！
如需在学术或商业用途中使用本项目，请注明出处。

---

## Table of Contents / 目录

* [Overview / 概述](#overview--概述)
* [Key Features / 核心功能](#key-features--核心功能)
* [System Architecture / 系统架构](#system-architecture--系统架构)
* [Usage Guide / 使用指南](#usage-guide--使用指南)
* [Project Structure / 项目结构](#project-structure--项目结构)
* [License / 许可证](#license--许可证)

---

## Overview / 概述

A professional deep learning-based CAD model classification system with intuitive 3D visualization. Users can load pre-trained UV-Net models, classify STEP files, and inspect results with interactive feedback.

一个结合深度学习与3D可视化的专业CAD分类系统。用户可加载预训练的UV-Net模型，对STEP文件进行分类，并以交互方式查看分类结果与置信度。

---

## Key Features / 核心功能

| Feature          | Description                       | 功能描述                  |
| ---------------- | --------------------------------- | --------------------- |
| Bilingual UI     | Seamless EN/CN language switching | 无缝中英文界面切换             |
| Deep Learning    | UV-Net based CAD classification   | 基于UV-Net的CAD分类        |
| Model Support    | Supports PyTorch Lightning models | 支持PyTorch Lightning模型 |
| Interactive View | Real-time 3D visualization        | 实时交互3D可视化             |
| Drag & Drop      | Easy file loading via drag & drop | 支持拖拽加载STEP文件          |


---

## System Architecture / 系统架构

The platform consists of three key modules:

1. **Preprocessing Module** - Normalizes CAD models and prepares input
2. **Classification Module** - UV-Net-based deep learning classifier
3. **UI Application** - PyQt5 GUI integrated with PythonOCC viewer

平台由三大模块构成：

1. **预处理模块** - 模型归一化与输入准备
2. **分类模块** - 基于UV-Net的深度学习分类器
3. **界面模块** - PyQt5前端与PythonOCC三维查看器

---

## Usage Guide / 使用指南

### Basic Workflow / 基本流程

1. **Load Model** - Load `.ckpt`, `.pt` or `.pth` file
   **加载模型** - 支持`.ckpt`、`.pt`、`.pth`格式
2. **Load Labels** - Load label mapping in `.json`
   **加载标签** - 支持`.json`标签映射
3. **Import CAD** - Drag & drop or manually load STEP file
   **导入模型** - 拖拽或手动加载STEP模型
4. **Classify** - System runs model and returns label
   **执行分类** - 系统运行模型并返回标签
5. **View Results** - See label, score and preview
   **查看结果** - 查看类别、置信度与模型预览

### Advanced Features / 高级功能

* **View Mode**: Shaded, wireframe, transparent
  **视图模式**: 着色、线框、透明切换
* **Export**: Save results, screenshots
  **导出**: 保存结果与模型截图
* **Replay**: Revisit past classification records
  **重放**: 浏览历史分类记录

---

## Project Structure / 项目结构

```
├── preprocessor.py        # 模型预处理与归一化
├── feature_extractor.py   # UV-Net特征提取与分类器
├── ui_app.py              # 主应用程序界面
├── README.md              # 项目说明文档
└── ui_app.spec            # PyInstaller 打包配置文件
```

---

## License / 许可证

MIT License

---

> For technical support or commercial inquiries, please contact the development team.
> 如需技术支持或商业合作，请联系开发团队。

---

