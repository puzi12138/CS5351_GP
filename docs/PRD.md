# 项目需求文档 (Project Requirements Document)

## 概述 (Overview)

拖拽式机器学习训练工具是一个基于Web的应用程序，旨在通过可视化编程方式简化机器学习模型的构建和训练过程。该工具面向非数据科学背景的开发者和学习者，使他们能够直观地理解和应用机器学习技术。

The Drag-and-Drop Machine Learning Training Tool is a web-based application designed to simplify the process of building and training machine learning models through visual programming. This tool is targeted at developers and learners without a data science background, enabling them to understand and apply machine learning techniques intuitively.

## 目标用户 (Target Users)

- 初学者和学生：希望学习机器学习基础知识的人
- 非数据科学背景的开发者：需要在项目中应用机器学习但缺乏专业知识的开发人员
- 教育工作者：需要教学工具来演示机器学习概念的教师
- 原型设计师：需要快速验证机器学习想法的设计师

- Beginners and students: People who want to learn the basics of machine learning
- Developers without a data science background: Developers who need to apply machine learning in their projects but lack specialized knowledge
- Educators: Teachers who need teaching tools to demonstrate machine learning concepts
- Prototype designers: Designers who need to quickly validate machine learning ideas

## 功能需求 (Functional Requirements)

### 1. 数据处理模块 (Data Processing Module)

- **数据集加载 (Dataset Loading)**
  - 支持加载预定义数据集（如Iris、MNIST等）
  - 支持导入自定义CSV数据
  - 提供数据预览功能

- **数据预处理 (Data Preprocessing)**
  - 数据分割（训练集/测试集）
  - 数据归一化/标准化
  - 特征选择

- **Dataset Loading**
  - Support for loading predefined datasets (such as Iris, MNIST, etc.)
  - Support for importing custom CSV data
  - Data preview functionality

- **Data Preprocessing**
  - Data splitting (training/test sets)
  - Data normalization/standardization
  - Feature selection

### 2. 模型构建模块 (Model Building Module)

- **模型选择 (Model Selection)**
  - 线性模型（线性回归、逻辑回归等）
  - 神经网络模型
  - 决策树模型

- **模型配置 (Model Configuration)**
  - 设置模型参数
  - 添加神经网络层
  - 配置激活函数

- **Model Selection**
  - Linear models (linear regression, logistic regression, etc.)
  - Neural network models
  - Decision tree models

- **Model Configuration**
  - Setting model parameters
  - Adding neural network layers
  - Configuring activation functions

### 3. 训练与评估模块 (Training and Evaluation Module)

- **模型训练 (Model Training)**
  - 设置训练参数（学习率、批量大小、迭代次数等）
  - 训练进度可视化
  - 训练过程中断/恢复

- **模型评估 (Model Evaluation)**
  - 准确率、精确率、召回率等指标计算
  - 混淆矩阵可视化
  - 学习曲线可视化

- **预测功能 (Prediction)**
  - 使用训练好的模型进行预测
  - 预测结果可视化

- **Model Training**
  - Setting training parameters (learning rate, batch size, number of iterations, etc.)
  - Training progress visualization
  - Training process interruption/resumption

- **Model Evaluation**
  - Calculation of metrics such as accuracy, precision, recall, etc.
  - Confusion matrix visualization
  - Learning curve visualization

- **Prediction**
  - Making predictions using the trained model
  - Prediction result visualization

### 4. 可视化模块 (Visualization Module)

- **数据可视化 (Data Visualization)**
  - 散点图、直方图、箱线图等
  - 特征相关性可视化

- **模型性能可视化 (Model Performance Visualization)**
  - 损失函数曲线
  - 准确率曲线
  - ROC曲线

- **Data Visualization**
  - Scatter plots, histograms, box plots, etc.
  - Feature correlation visualization

- **Model Performance Visualization**
  - Loss function curves
  - Accuracy curves
  - ROC curves

### 5. 项目管理功能 (Project Management Features)

- **保存/加载项目 (Save/Load Projects)**
  - 保存当前工作区状态
  - 加载已保存的项目

- **代码导出 (Code Export)**
  - 导出生成的Python/JavaScript代码
  - 代码复制功能

- **Save/Load Projects**
  - Save current workspace state
  - Load saved projects

- **Code Export**
  - Export generated Python/JavaScript code
  - Code copying functionality

## 非功能需求 (Non-functional Requirements)

### 1. 性能 (Performance)

- 界面响应时间不超过1秒
- 小型数据集的训练过程在浏览器中流畅运行
- 支持渐进式加载大型数据集

- Interface response time not exceeding 1 second
- Smooth running of training process for small datasets in the browser
- Support for progressive loading of large datasets

### 2. 可用性 (Usability)

- 直观的拖拽界面
- 详细的工具提示和帮助文档
- 适合初学者的用户界面
- 响应式设计，支持不同屏幕尺寸

- Intuitive drag-and-drop interface
- Detailed tooltips and help documentation
- Beginner-friendly user interface
- Responsive design, supporting different screen sizes

### 3. 兼容性 (Compatibility)

- 支持主流浏览器（Chrome、Firefox、Safari、Edge）
- 支持桌面和平板设备

- Support for mainstream browsers (Chrome, Firefox, Safari, Edge)
- Support for desktop and tablet devices

### 4. 安全性 (Security)

- 用户数据不离开本地浏览器
- 无需后端服务器处理敏感数据

- User data does not leave the local browser
- No backend server required to process sensitive data

## 用户故事 (User Stories)

1. 作为一名学生，我希望能够通过拖拽方式构建简单的机器学习模型，以便我能够理解机器学习的基本概念。
2. 作为一名Web开发者，我希望能够快速训练一个图像分类模型，而不需要深入了解TensorFlow的复杂API。
3. 作为一名教师，我希望能够在课堂上演示机器学习模型的训练过程，以便学生能够直观地理解学习算法的工作原理。
4. 作为一名产品经理，我希望能够快速验证数据集是否适合机器学习应用，而不需要编写代码。

1. As a student, I want to be able to build simple machine learning models through drag-and-drop, so that I can understand the basic concepts of machine learning.
2. As a web developer, I want to quickly train an image classification model without having to delve into TensorFlow's complex API.
3. As a teacher, I want to be able to demonstrate the training process of machine learning models in the classroom, so that students can intuitively understand how learning algorithms work.
4. As a product manager, I want to quickly validate whether a dataset is suitable for machine learning applications without having to write code.

## 验收标准 (Acceptance Criteria)

1. 用户能够成功加载预定义数据集并进行预处理
2. 用户能够通过拖拽方式构建至少三种不同类型的机器学习模型
3. 模型训练过程中显示实时进度和性能指标
4. 训练完成后能够可视化模型性能并进行预测
5. 用户能够保存工作区状态并在之后恢复
6. 系统能够生成与拖拽块对应的代码
7. 所有主要功能在Chrome和Firefox浏览器上正常工作

1. Users can successfully load predefined datasets and preprocess them
2. Users can build at least three different types of machine learning models through drag-and-drop
3. Real-time progress and performance metrics are displayed during model training
4. After training, model performance can be visualized and predictions can be made
5. Users can save workspace state and restore it later
6. The system can generate code corresponding to the drag-and-drop blocks
7. All major functions work properly on Chrome and Firefox browsers

## 未来扩展 (Future Extensions)

- 支持更多类型的机器学习模型（如强化学习、推荐系统等）
- 添加协作功能，允许多用户同时编辑项目
- 开发模型部署功能，允许用户将训练好的模型导出为可部署格式
- 添加更多高级数据预处理选项
- 支持更复杂的神经网络架构（如CNN、RNN等）

- Support for more types of machine learning models (such as reinforcement learning, recommendation systems, etc.)
- Add collaboration features, allowing multiple users to edit projects simultaneously
- Develop model deployment functionality, allowing users to export trained models in deployable formats
- Add more advanced data preprocessing options
- Support for more complex neural network architectures (such as CNN, RNN, etc.)