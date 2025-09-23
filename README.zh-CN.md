<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本项目是一个基于 **Streamlit** 的交互式应用，用于学习和练习 **多层感知器（MLP, Multi-Layer Perceptron）**
的基本原理和训练流程。用户可以通过可视化界面上传数据、调整模型参数、训练模型，并实时查看训练结果与预测效果。  
该应用主要面向希望了解神经网络基础、分类与回归问题建模流程的学生和初学者。通过交互操作，用户可以更直观地理解 MLP
的结构、激活函数、损失函数以及优化器的作用。用户可以：

- 加载并预处理 MNIST 数据集。
- 训练可定制的 MLP 模型，并实时查看训练指标。
- 在测试集上测试模型性能。
- 在画布上手写数字并获得即时预测结果。

系统旨在帮助初学者和深度学习爱好者以交互和可视化的方式探索模型训练与预测过程。

**数据描述**
---
本应用内置 MNIST 手写数字数据集，适合监督学习练习：

1. MNIST 手写数字

+ 特征：28×28 灰度图像，已展平为 784 维向量（X_train_flat / X_test_flat）
+ 标签：数字类别 0–9（y_train / y_test）
+ 用途：练习 MLP 分类任务，观察训练动态，评估模型性能，并可可视化预测结果。

2. 自定义数据上传（可选）

+ 用户可上传 CSV 数据进行训练，要求遵循 特征列 + 标签列 结构。

**功能特性**
---

- **数据加载与预处理：** 加载 MNIST 数据集并进行扁平化、归一化处理。
- **模型训练：** 支持自定义训练轮数、批量大小和验证集比例的多层感知机训练。
- **实时训练指标：** 可实时监控训练与验证集的损失、准确率、精确率、召回率和 AUC。
- **模型测试：** 在测试集上评估模型性能，提供准确率和 R² 值。
- **实时数字识别：** 在画布上绘制数字并使用训练好的模型进行即时预测。
- **可视化工具：** 支持二维/三维散点图与决策边界可视化（可用于 MNIST 以外的实验数据）。

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://mlp-mnist.streamlit.app/)

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause License](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
