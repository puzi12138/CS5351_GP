# 前端开发指南 (Frontend Development Guide)

## 概述 (Overview)

本文档提供拖拽式机器学习训练工具前端开发的详细指南，包括UI组件、交互设计、国际化等方面。该文档旨在帮助开发者理解和扩展前端功能。

This document provides a detailed guide for frontend development of the Drag-and-Drop Machine Learning Training Tool, including UI components, interaction design, internationalization, etc. The document aims to help developers understand and extend frontend functionality.

## 技术栈 (Technology Stack)

### 核心库 (Core Libraries)

- **Blockly**: 提供拖拽式编程界面
- **Bootstrap 4.5.3**: 提供响应式UI组件和布局
- **jQuery 3.5.1**: DOM操作和事件处理
- **Chart.js 2.9.4**: 数据可视化
- **ONNX.js**: 支持在浏览器中运行PyTorch模型

- **Blockly**: Provides drag-and-drop programming interface
- **Bootstrap 4.5.3**: Provides responsive UI components and layouts
- **jQuery 3.5.1**: DOM manipulation and event handling
- **Chart.js 2.9.4**: Data visualization
- **ONNX.js**: Supports running PyTorch models in the browser

## 项目结构 (Project Structure)

```
/
├── index.html          # 主HTML文件
├── static/
│   ├── css/
│   │   └── style.css   # 自定义样式
│   └── js/
│       ├── app.js      # 应用程序主逻辑
│       ├── blocks.js   # Blockly块定义
│       └── generators.js # 代码生成器
└── docs/               # 文档
```

## UI组件 (UI Components)

### 导航栏 (Navigation Bar)

导航栏位于页面顶部，包含以下菜单：

- **File**: 包含新建项目、保存项目、加载项目等选项
- **View**: 包含查看代码、查看数据等选项
- **Run**: 包含运行模型、停止执行等选项
- **Help**: 显示帮助信息

The navigation bar is located at the top of the page and contains the following menus:

- **File**: Contains options such as new project, save project, load project, etc.
- **View**: Contains options such as view code, view data, etc.
- **Run**: Contains options such as run model, stop execution, etc.
- **Help**: Displays help information

### 工作区布局 (Workspace Layout)

页面主体分为三个部分：

1. **左侧工具箱 (Left Toolbox)**: 包含可拖拽的Blockly块，分为数据处理、模型构建、训练与评估、可视化四类
2. **中间Blockly工作区 (Middle Blockly Workspace)**: 用户拖拽和连接块的主要区域
3. **右侧输出区域 (Right Output Area)**: 显示控制台输出、可视化结果和生成的代码

The main body of the page is divided into three parts:

1. **Left Toolbox**: Contains draggable Blockly blocks, divided into four categories: data processing, model building, training and evaluation, visualization
2. **Middle Blockly Workspace**: The main area for users to drag and connect blocks
3. **Right Output Area**: Displays console output, visualization results, and generated code

### 模态框 (Modal Dialogs)

- **代码模态框 (Code Modal)**: 显示生成的Python代码，提供复制功能
- **帮助模态框 (Help Modal)**: 显示基本操作和块说明的帮助信息

- **Code Modal**: Displays the generated Python code, provides copy functionality
- **Help Modal**: Displays help information about basic operations and block descriptions

## Blockly块设计 (Blockly Block Design)

### 块类别 (Block Categories)

1. **数据处理 (Data Processing)**
   - 加载数据集 (Load Dataset)
   - 分割数据 (Split Data)
   - 归一化数据 (Normalize Data)

2. **模型构建 (Model Building)**
   - 创建线性模型 (Create Linear Model)
   - 创建神经网络 (Create Neural Network)
   - 添加层 (Add Layer)

3. **训练与评估 (Training & Evaluation)**
   - 训练模型 (Train Model)
   - 评估模型 (Evaluate Model)
   - 预测 (Predict)

4. **可视化 (Visualization)**
   - 绘制数据 (Plot Data)
   - 绘制训练历史 (Plot History)
   - 混淆矩阵 (Confusion Matrix)

### 块设计原则 (Block Design Principles)

- **一致性 (Consistency)**: 保持块的外观和行为一致
- **自解释性 (Self-explanatory)**: 块的名称和字段应清晰表达其功能
- **适当的连接点 (Appropriate Connection Points)**: 确保块之间的连接逻辑合理
- **合理的默认值 (Reasonable Default Values)**: 为字段提供合理的默认值

- **Consistency**: Maintain consistent appearance and behavior of blocks
- **Self-explanatory**: Block names and fields should clearly express their functionality
- **Appropriate Connection Points**: Ensure logical connections between blocks
- **Reasonable Default Values**: Provide reasonable default values for fields

## 交互设计 (Interaction Design)

### 拖拽操作 (Drag and Drop Operations)

- 从左侧工具箱拖拽块到中间工作区
- 在工作区内拖动块调整位置
- 连接块形成完整的机器学习流程

- Drag blocks from the left toolbox to the middle workspace
- Drag blocks within the workspace to adjust positions
- Connect blocks to form a complete machine learning workflow

### 运行与反馈 (Execution and Feedback)

- 点击运行按钮执行整个流程
- 在右侧控制台实时显示执行进度和结果
- 在可视化区域显示图表和模型性能

- Click the run button to execute the entire workflow
- Display execution progress and results in real-time in the right console
- Display charts and model performance in the visualization area

### 错误处理 (Error Handling)

- 在控制台显示错误信息
- 高亮显示出错的块
- 提供错误修复建议

- Display error messages in the console
- Highlight blocks with errors
- Provide error correction suggestions

## 国际化 (Internationalization)

### 支持的语言 (Supported Languages)

- 英语 (English)
- 中文 (Chinese)

### 国际化实现 (Internationalization Implementation)

- 使用Blockly内置的国际化支持
- 所有UI文本使用国际化字符串
- 动态加载语言文件

- Use Blockly's built-in internationalization support
- All UI text uses internationalized strings
- Dynamically load language files

### 添加新语言 (Adding New Languages)

1. 创建新的语言文件
2. 翻译所有字符串
3. 在应用程序中注册新语言

1. Create a new language file
2. Translate all strings
3. Register the new language in the application

## 代码生成 (Code Generation)

### 生成的代码类型 (Generated Code Types)

- **JavaScript**: 用于在浏览器中执行
- **Python**: 用于导出和在本地环境执行

- **JavaScript**: For execution in the browser
- **Python**: For export and execution in local environments

### 代码生成流程 (Code Generation Process)

1. 遍历工作区中的块
2. 为每个块生成相应的代码
3. 组合代码形成完整的程序

1. Traverse blocks in the workspace
2. Generate corresponding code for each block
3. Combine code to form a complete program

### 自定义代码生成器 (Custom Code Generators)

- 在`generators.js`中定义每个块的代码生成函数
- 使用Blockly的代码生成API
- 支持多种编程语言

- Define code generation functions for each block in `generators.js`
- Use Blockly's code generation API
- Support multiple programming languages

## 扩展指南 (Extension Guide)

### 添加新的块 (Adding New Blocks)

1. 在`blocks.js`中定义新块的外观和行为
2. 在`generators.js`中添加相应的代码生成函数
3. 在工具箱中注册新块

1. Define the appearance and behavior of new blocks in `blocks.js`
2. Add corresponding code generation functions in `generators.js`
3. Register new blocks in the toolbox

### 添加新的功能 (Adding New Features)

1. 在`app.js`中添加新的功能函数
2. 更新UI以支持新功能
3. 添加相应的事件处理程序

1. Add new feature functions in `app.js`
2. Update the UI to support new features
3. Add corresponding event handlers

### 添加新的可视化 (Adding New Visualizations)

1. 使用Chart.js创建新的图表类型
2. 在`app.js`中添加相应的可视化函数
3. 更新块和代码生成器以支持新的可视化

1. Create new chart types using Chart.js
2. Add corresponding visualization functions in `app.js`
3. Update blocks and code generators to support new visualizations

## 性能优化 (Performance Optimization)

### 大型数据集处理 (Large Dataset Processing)

- 使用分批处理减少内存使用
- 实现数据流式加载
- 优化数据结构减少内存占用

- Use batch processing to reduce memory usage
- Implement streaming data loading
- Optimize data structures to reduce memory footprint

### 渲染优化 (Rendering Optimization)

- 减少DOM操作
- 使用requestAnimationFrame进行动画
- 优化Chart.js渲染性能

- Reduce DOM operations
- Use requestAnimationFrame for animations
- Optimize Chart.js rendering performance

### 代码执行优化 (Code Execution Optimization)

- 使用Web Workers进行后台计算
- 实现计算结果缓存
- 优化算法减少计算复杂度

- Use Web Workers for background computation
- Implement computation result caching
- Optimize algorithms to reduce computational complexity

## 调试指南 (Debugging Guide)

### 浏览器开发工具 (Browser Developer Tools)

- 使用控制台查看日志和错误
- 使用网络面板监控数据加载
- 使用性能面板分析性能瓶颈

- Use the console to view logs and errors
- Use the network panel to monitor data loading
- Use the performance panel to analyze performance bottlenecks

### 应用程序日志 (Application Logs)

- 在控制台输出区域查看应用程序日志
- 使用不同的日志级别（信息、警告、错误）
- 记录关键操作和状态变化

- View application logs in the console output area
- Use different log levels (info, warning, error)
- Record key operations and state changes

### 常见问题排查 (Common Issue Troubleshooting)

- 块连接问题
- 数据加载错误
- 模型训练失败
- 可视化显示异常

- Block connection issues
- Data loading errors
- Model training failures
- Visualization display anomalies