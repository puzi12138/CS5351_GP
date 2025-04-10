# 应用流程文档 (Application Flow Document)

## 概述 (Overview)

本文档详细说明拖拽式机器学习训练工具的使用流程，包括数据加载、预处理、模型构建、训练、评估和可视化等步骤。该文档旨在帮助用户理解应用程序的工作流程和操作方法。

This document details the workflow of the Drag-and-Drop Machine Learning Training Tool, including data loading, preprocessing, model building, training, evaluation, and visualization steps. The document aims to help users understand the application's workflow and operation methods.

## 基本工作流程 (Basic Workflow)

### 1. 创建新项目 (Create New Project)

- 点击导航栏中的 "File" -> "New Project" 创建一个新的工作区
- 工作区将被清空，准备开始新的机器学习项目

- Click "File" -> "New Project" in the navigation bar to create a new workspace
- The workspace will be cleared, ready to start a new machine learning project

### 2. 数据处理 (Data Processing)

#### 2.1 加载数据集 (Load Dataset)

- 从左侧工具箱中拖拽 "Load Dataset" 块到工作区
- 选择预定义数据集（如Iris、MNIST等）或上传自定义CSV文件
- 为数据集指定变量名

- Drag the "Load Dataset" block from the left toolbox to the workspace
- Select a predefined dataset (such as Iris, MNIST, etc.) or upload a custom CSV file
- Specify a variable name for the dataset

#### 2.2 数据分割 (Split Data)

- 拖拽 "Split Data" 块到工作区并连接到数据集块
- 设置测试集比例（如0.2，表示20%的数据用于测试）
- 设置随机种子以确保结果可重现
- 指定变量名保存分割后的数据（通常为X_train, X_test, y_train, y_test）

- Drag the "Split Data" block to the workspace and connect it to the dataset block
- Set the test set ratio (e.g., 0.2, indicating 20% of the data is used for testing)
- Set a random seed to ensure reproducible results
- Specify variable names to save the split data (typically X_train, X_test, y_train, y_test)

#### 2.3 数据归一化 (Normalize Data)

- 拖拽 "Normalize Data" 块到工作区
- 选择归一化方法（标准化、最小最大缩放等）
- 连接到需要归一化的数据
- 指定变量名保存归一化后的数据

- Drag the "Normalize Data" block to the workspace
- Select a normalization method (standardization, min-max scaling, etc.)
- Connect to the data that needs to be normalized
- Specify a variable name to save the normalized data

### 3. 模型构建 (Model Building)

#### 3.1 创建模型 (Create Model)

- 从左侧工具箱中拖拽模型创建块（如 "Create Linear Model" 或 "Create Neural Network"）
- 配置模型参数（如模型类型、输入维度等）
- 指定变量名保存创建的模型

- Drag a model creation block (such as "Create Linear Model" or "Create Neural Network") from the left toolbox
- Configure model parameters (such as model type, input dimensions, etc.)
- Specify a variable name to save the created model

#### 3.2 添加神经网络层 (Add Neural Network Layers)

- 如果创建的是神经网络模型，可以拖拽 "Add Layer" 块添加网络层
- 选择层类型（Dense、Dropout、Conv2D等）
- 设置层参数（神经元数量、激活函数等）
- 连接到模型块

- If a neural network model is created, you can drag the "Add Layer" block to add network layers
- Select the layer type (Dense, Dropout, Conv2D, etc.)
- Set layer parameters (number of neurons, activation function, etc.)
- Connect to the model block

### 4. 模型训练 (Model Training)

- 拖拽 "Train Model" 块到工作区
- 连接到模型块
- 设置训练参数（批次大小、训练轮数等）
- 连接训练数据（X_train, y_train）
- 指定变量名保存训练历史

- Drag the "Train Model" block to the workspace
- Connect to the model block
- Set training parameters (batch size, number of epochs, etc.)
- Connect training data (X_train, y_train)
- Specify a variable name to save the training history

### 5. 模型评估 (Model Evaluation)

- 拖拽 "Evaluate Model" 块到工作区
- 连接到训练好的模型
- 连接测试数据（X_test, y_test）
- 查看评估指标（准确率、损失值等）

- Drag the "Evaluate Model" block to the workspace
- Connect to the trained model
- Connect test data (X_test, y_test)
- View evaluation metrics (accuracy, loss value, etc.)

### 6. 可视化 (Visualization)

#### 6.1 数据可视化 (Data Visualization)

- 拖拽 "Plot Data" 块到工作区
- 选择可视化类型（散点图、直方图等）
- 连接到要可视化的数据
- 设置可视化参数（标题、轴标签等）

- Drag the "Plot Data" block to the workspace
- Select the visualization type (scatter plot, histogram, etc.)
- Connect to the data to be visualized
- Set visualization parameters (title, axis labels, etc.)

#### 6.2 训练历史可视化 (Training History Visualization)

- 拖拽 "Plot History" 块到工作区
- 连接到训练历史变量
- 选择要可视化的指标（损失值、准确率等）

- Drag the "Plot History" block to the workspace
- Connect to the training history variable
- Select metrics to visualize (loss value, accuracy, etc.)

#### 6.3 混淆矩阵可视化 (Confusion Matrix Visualization)

- 拖拽 "Confusion Matrix" 块到工作区
- 连接到模型和测试数据
- 查看分类结果的混淆矩阵

- Drag the "Confusion Matrix" block to the workspace
- Connect to the model and test data
- View the confusion matrix of classification results

### 7. 预测 (Prediction)

- 拖拽 "Predict" 块到工作区
- 连接到训练好的模型
- 连接要进行预测的数据
- 查看预测结果

- Drag the "Predict" block to the workspace
- Connect to the trained model
- Connect to the data to be predicted
- View prediction results

### 8. 保存项目 (Save Project)

- 点击导航栏中的 "File" -> "Save Project"
- 输入项目名称
- 项目将被保存，可以在之后加载

- Click "File" -> "Save Project" in the navigation bar
- Enter a project name
- The project will be saved and can be loaded later

## 示例流程 (Example Workflow)

### 鸢尾花分类示例 (Iris Classification Example)

1. 创建新项目
2. 加载Iris数据集
3. 将数据分割为训练集和测试集（比例0.8:0.2）
4. 创建逻辑回归模型
5. 使用训练数据训练模型（10轮）
6. 使用测试数据评估模型
7. 可视化混淆矩阵
8. 保存项目

1. Create a new project
2. Load the Iris dataset
3. Split the data into training and test sets (ratio 0.8:0.2)
4. Create a logistic regression model
5. Train the model using training data (10 epochs)
6. Evaluate the model using test data
7. Visualize the confusion matrix
8. Save the project

### 手写数字识别示例 (MNIST Digit Recognition Example)

1. 创建新项目
2. 加载MNIST数据集
3. 将数据分割为训练集和测试集（比例0.8:0.2）
4. 创建神经网络模型
5. 添加Dense层（128个神经元，ReLU激活）
6. 添加Dropout层（比例0.2）
7. 添加Dense层（10个神经元，Softmax激活）
8. 使用训练数据训练模型（5轮）
9. 使用测试数据评估模型
10. 可视化训练历史（准确率和损失值）
11. 可视化混淆矩阵
12. 保存项目

1. Create a new project
2. Load the MNIST dataset
3. Split the data into training and test sets (ratio 0.8:0.2)
4. Create a neural network model
5. Add a Dense layer (128 neurons, ReLU activation)
6. Add a Dropout layer (ratio 0.2)
7. Add a Dense layer (10 neurons, Softmax activation)
8. Train the model using training data (5 epochs)
9. Evaluate the model using test data
10. Visualize training history (accuracy and loss value)
11. Visualize the confusion matrix
12. Save the project

## 常见问题解答 (FAQ)

### Q: 如何加载自定义数据集？
**A**: 在 "Load Dataset" 块中选择 "自定义CSV"，然后点击运行后会弹出文件选择对话框，选择您的CSV文件即可。

### Q: How to load a custom dataset?
**A**: Select "Custom CSV" in the "Load Dataset" block, then click run and a file selection dialog will pop up, select your CSV file.

### Q: 如何查看生成的代码？
**A**: 点击导航栏中的 "View" -> "View Code" 或切换到右侧输出区域的 "Code" 标签页。

### Q: How to view the generated code?
**A**: Click "View" -> "View Code" in the navigation bar or switch to the "Code" tab in the right output area.

### Q: 如何中断正在运行的模型训练？
**A**: 点击导航栏中的 "Run" -> "Stop Execution"。

### Q: How to interrupt a running model training?
**A**: Click "Run" -> "Stop Execution" in the navigation bar.