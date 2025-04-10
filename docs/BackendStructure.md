# 后端结构文档 (Backend Structure Document)

## 概述 (Overview)

本文档详细说明拖拽式机器学习训练工具的后端结构，包括PyTorch集成、数据处理流程和模型训练实现等。该文档旨在帮助开发者理解系统的后端设计和机器学习模型的实现方式。

This document details the backend structure of the Drag-and-Drop Machine Learning Training Tool, including PyTorch integration, data processing flow, and model training implementation. The document aims to help developers understand the backend design of the system and the implementation of machine learning models.

## 架构设计 (Architecture Design)

### 客户端架构 (Client-side Architecture)

本应用采用纯前端架构，所有计算和处理都在浏览器中进行，无需后端服务器。这种设计有以下优势：

- **隐私保护**: 用户数据不离开本地浏览器
- **易于部署**: 无需配置服务器环境
- **离线工作**: 加载后可在无网络环境下使用

The application adopts a pure frontend architecture, with all computation and processing done in the browser without requiring a backend server. This design has the following advantages:

- **Privacy Protection**: User data does not leave the local browser
- **Easy Deployment**: No need to configure server environments
- **Offline Work**: Can be used in offline environments after loading

### PyTorch集成 (PyTorch Integration)

虽然PyTorch主要设计用于Python环境，但我们通过以下方式在浏览器中集成PyTorch功能：

1. **ONNX.js**: 使用ONNX.js加载预先转换为ONNX格式的PyTorch模型
2. **模型转换**: 提供工具将PyTorch模型转换为ONNX格式
3. **JavaScript API**: 实现类似PyTorch的JavaScript API，保持概念一致性

Although PyTorch is primarily designed for Python environments, we integrate PyTorch functionality in the browser through the following methods:

1. **ONNX.js**: Use ONNX.js to load PyTorch models pre-converted to ONNX format
2. **Model Conversion**: Provide tools to convert PyTorch models to ONNX format
3. **JavaScript API**: Implement JavaScript APIs similar to PyTorch, maintaining conceptual consistency

## 数据处理 (Data Processing)

### 数据加载 (Data Loading)

#### 预定义数据集 (Predefined Datasets)

系统支持以下预定义数据集：

- **Iris**: 鸢尾花分类数据集
- **Boston Housing**: 波士顿房价预测数据集
- **MNIST**: 手写数字识别数据集
- **Fashion MNIST**: 时尚物品分类数据集
- **CIFAR10**: 彩色图像分类数据集

预定义数据集通过CDN或静态文件提供，使用fetch API加载。

The system supports the following predefined datasets:

- **Iris**: Iris flower classification dataset
- **Boston Housing**: Boston housing price prediction dataset
- **MNIST**: Handwritten digit recognition dataset
- **Fashion MNIST**: Fashion item classification dataset
- **CIFAR10**: Color image classification dataset

Predefined datasets are provided through CDN or static files and loaded using the fetch API.

#### 自定义CSV数据 (Custom CSV Data)

用户可以上传自定义CSV文件，系统会：

1. 使用FileReader API读取文件内容
2. 解析CSV格式，提取表头和数据行
3. 将字符串数据转换为数值型数据
4. 自动识别特征列和标签列

Users can upload custom CSV files, and the system will:

1. Use the FileReader API to read file contents
2. Parse the CSV format, extracting headers and data rows
3. Convert string data to numeric data
4. Automatically identify feature columns and label columns

### 数据预处理 (Data Preprocessing)

#### 数据分割 (Data Splitting)

实现训练集和测试集分割：

```javascript
function splitData(data, testSize = 0.2, randomState = 42) {
  // 设置随机种子
  Math.seedrandom(randomState);
  
  // 随机打乱数据
  const shuffled = [...data];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  
  // 分割数据
  const testCount = Math.floor(shuffled.length * testSize);
  const testData = shuffled.slice(0, testCount);
  const trainData = shuffled.slice(testCount);
  
  return {
    trainData,
    testData
  };
}
```

Implementation of training and test set splitting:

```javascript
function splitData(data, testSize = 0.2, randomState = 42) {
  // Set random seed
  Math.seedrandom(randomState);
  
  // Randomly shuffle data
  const shuffled = [...data];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  
  // Split data
  const testCount = Math.floor(shuffled.length * testSize);
  const testData = shuffled.slice(0, testCount);
  const trainData = shuffled.slice(testCount);
  
  return {
    trainData,
    testData
  };
}
```

#### 数据归一化 (Data Normalization)

支持以下归一化方法：

1. **标准化 (StandardScaler)**:
   ```javascript
   function standardize(data) {
     const mean = calculateMean(data);
     const std = calculateStd(data);
     return data.map(row => row.map((val, i) => (val - mean[i]) / std[i]));
   }
   ```

2. **最小最大缩放 (MinMaxScaler)**:
   ```javascript
   function minMaxScale(data) {
     const min = calculateMin(data);
     const max = calculateMax(data);
     return data.map(row => row.map((val, i) => 
       (val - min[i]) / (max[i] - min[i])
     ));
   }
   ```

3. **鲁棒缩放 (RobustScaler)**:
   ```javascript
   function robustScale(data) {
     const median = calculateMedian(data);
     const iqr = calculateIQR(data);
     return data.map(row => row.map((val, i) => 
       (val - median[i]) / iqr[i]
     ));
   }
   ```

Supports the following normalization methods:

1. **Standardization (StandardScaler)**:
   ```javascript
   function standardize(data) {
     const mean = calculateMean(data);
     const std = calculateStd(data);
     return data.map(row => row.map((val, i) => (val - mean[i]) / std[i]));
   }
   ```

2. **Min-Max Scaling (MinMaxScaler)**:
   ```javascript
   function minMaxScale(data) {
     const min = calculateMin(data);
     const max = calculateMax(data);
     return data.map(row => row.map((val, i) => 
       (val - min[i]) / (max[i] - min[i])
     ));
   }
   ```

3. **Robust Scaling (RobustScaler)**:
   ```javascript
   function robustScale(data) {
     const median = calculateMedian(data);
     const iqr = calculateIQR(data);
     return data.map(row => row.map((val, i) => 
       (val - median[i]) / iqr[i]
     ));
   }
   ```

## 模型实现 (Model Implementation)

### 线性模型 (Linear Models)

#### 线性回归 (Linear Regression)

```javascript
class LinearRegression {
  constructor(learningRate = 0.01, iterations = 1000) {
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.weights = null;
    this.bias = null;
  }
  
  fit(X, y) {
    // 初始化参数
    const n_samples = X.length;
    const n_features = X[0].length;
    this.weights = Array(n_features).fill(0);
    this.bias = 0;
    
    // 梯度下降
    for (let i = 0; i < this.iterations; i++) {
      // 预测
      const y_pred = X.map(x => this._predict_row(x));
      
      // 计算梯度
      const dw = Array(n_features).fill(0);
      let db = 0;
      
      for (let j = 0; j < n_samples; j++) {
        const error = y_pred[j] - y[j];
        
        for (let k = 0; k < n_features; k++) {
          dw[k] += error * X[j][k];
        }
        
        db += error;
      }
      
      // 更新参数
      for (let k = 0; k < n_features; k++) {
        this.weights[k] -= (this.learningRate * dw[k]) / n_samples;
      }
      
      this.bias -= (this.learningRate * db) / n_samples;
    }
    
    return this;
  }
  
  predict(X) {
    return X.map(x => this._predict_row(x));
  }
  
  _predict_row(x) {
    let y_pred = this.bias;
    
    for (let i = 0; i < x.length; i++) {
      y_pred += this.weights[i] * x[i];
    }
    
    return y_pred;
  }
}
```

#### 逻辑回归 (Logistic Regression)

```javascript
class LogisticRegression extends LinearRegression {
  constructor(learningRate = 0.01, iterations = 1000) {
    super(learningRate, iterations);
  }
  
  _sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }
  
  _predict_row(x) {
    const linear_pred = super._predict_row(x);
    return this._sigmoid(linear_pred);
  }
  
  fit(X, y) {
    // 初始化参数
    const n_samples = X.length;
    const n_features = X[0].length;
    this.weights = Array(n_features).fill(0);
    this.bias = 0;
    
    // 梯度下降
    for (let i = 0; i < this.iterations; i++) {
      // 预测
      const y_pred = X.map(x => this._predict_row(x));
      
      // 计算梯度
      const dw = Array(n_features).fill(0);
      let db = 0;
      
      for (let j = 0; j < n_samples; j++) {
        const error = y_pred[j] - y[j];
        
        for (let k = 0; k < n_features; k++) {
          dw[k] += error * X[j][k];
        }
        
        db += error;
      }
      
      // 更新参数
      for (let k = 0; k < n_features; k++) {
        this.weights[k] -= (this.learningRate * dw[k]) / n_samples;
      }
      
      this.bias -= (this.learningRate * db) / n_samples;
    }
    
    return this;
  }
}
```

### 神经网络模型 (Neural Network Models)

#### 简单神经网络 (Simple Neural Network)

```javascript
class NeuralNetwork {
  constructor(inputDim) {
    this.layers = [];
    this.inputDim = inputDim;
  }
  
  addLayer(units, activation = 'relu', dropout = 0) {
    this.layers.push({
      units,
      activation,
      dropout,
      weights: null,
      bias: null
    });
    
    return this;
  }
  
  compile() {
    // 初始化权重和偏置
    let inputSize = this.inputDim;
    
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const outputSize = layer.units;
      
      // 初始化权重（使用Xavier初始化）
      const stddev = Math.sqrt(2 / (inputSize + outputSize));
      layer.weights = Array(outputSize).fill().map(() => 
        Array(inputSize).fill().map(() => 
          randomNormal(0, stddev)
        )
      );
      
      // 初始化偏置
      layer.bias = Array(outputSize).fill(0);
      
      // 更新下一层的输入大小
      inputSize = outputSize;
    }
    
    return this;
  }
  
  fit(X, y, epochs = 10, batchSize = 32, learningRate = 0.01) {
    if (!this.layers.length) {
      throw new Error('模型没有层，请先添加层');
    }
    
    if (!this.layers[0].weights) {
      this.compile();
    }
    
    const history = {
      loss: [],
      accuracy: []
    };
    
    const n_samples = X.length;
    const n_batches = Math.ceil(n_samples / batchSize);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let correctPredictions = 0;
      
      // 随机打乱数据
      const indices = Array(n_samples).fill().map((_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      // 批次训练
      for (let batch = 0; batch < n_batches; batch++) {
        const startIdx = batch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, n_samples);
        const batchIndices = indices.slice(startIdx, endIdx);
        
        const X_batch = batchIndices.map(i => X[i]);
        const y_batch = batchIndices.map(i => y[i]);
        
        // 前向传播
        const outputs = this._forwardPass(X_batch);
        
        // 计算损失
        const loss = this._computeLoss(outputs[outputs.length - 1], y_batch);
        totalLoss += loss;
        
        // 计算准确率
        const predictions = outputs[outputs.length - 1].map(row => 
          row.indexOf(Math.max(...row))
        );
        
        for (let i = 0; i < predictions.length; i++) {
          if (predictions[i] === y_batch[i]) {
            correctPredictions++;
          }
        }
        
        // 反向传播
        this._backwardPass(X_batch, y_batch, outputs, learningRate);
      }
      
      // 记录历史
      const epochLoss = totalLoss / n_batches;
      const epochAccuracy = correctPredictions / n_samples;
      
      history.loss.push(epochLoss);
      history.accuracy.push(epochAccuracy);
      
      console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${epochLoss.toFixed(4)}, Accuracy: ${epochAccuracy.toFixed(4)}`);
    }
    
    return history;
  }
  
  predict(X) {
    const outputs = this._forwardPass(X);
    return outputs[outputs.length - 1].map(row => 
      row.indexOf(Math.max(...row))
    );
  }
  
  _forwardPass(X) {
    const outputs = [X];
    let currentOutput = X;
    
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const nextOutput = [];
      
      for (let j = 0; j < currentOutput.length; j++) {
        const input = currentOutput[j];
        const output = Array(layer.units).fill(0);
        
        // 计算线性组合
        for (let k = 0; k < layer.units; k++) {
          output[k] = layer.bias[k];
          
          for (let l = 0; l < input.length; l++) {
            output[k] += layer.weights[k][l] * input[l];
          }
        }
        
        // 应用激活函数
        for (let k = 0; k < output.length; k++) {
          output[k] = this._activate(output[k], layer.activation);
        }
        
        // 应用Dropout（仅在训练时）
        if (layer.dropout > 0) {
          for (let k = 0; k < output.length; k++) {
            if (Math.random() < layer.dropout) {
              output[k] = 0;
            } else {
              output[k] /= (1 - layer.dropout); // 缩放以保持期望值不变
            }
          }
        }
        
        nextOutput.push(output);
      }
      
      currentOutput = nextOutput;
      outputs.push(currentOutput);
    }
    
    return outputs;
  }
  
  _activate(x, activation) {
    switch (activation) {
      case 'relu':
        return Math.max(0, x);
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x));
      case 'tanh':
        return Math.tanh(x);
      case 'softmax':
        // Softmax在层级应用，不是单个值
        return x;
      default:
        return x;
    }
  }
  
  _computeLoss(predictions, targets) {
    // 简化的交叉熵损失
    let loss = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const target = targets[i];
      const pred = predictions[i][target];
      loss -= Math.log(pred + 1e-10); // 添加小值避免log(0)
    }
    
    return loss / predictions.length;
  }
  
  _backwardPass(X, y, outputs, learningRate) {
    // 简化的反向传播实现
    const n_samples = X.length;
    
    // 计算输出层梯度
    const outputGradients = [];
    const outputLayer = outputs[outputs.length - 1];
    
    for (let i = 0; i < n_samples; i++) {
      const gradient = Array(outputLayer[i].length).fill(0);
      gradient[y[i]] = -1 / (outputLayer[i][y[i]] + 1e-10);
      outputGradients.push(gradient);
    }
    
    // 反向传播梯度
    let currentGradients = outputGradients;
    
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevOutput = outputs[l];
      const nextGradients = [];
      
      // 更新权重和偏置
      for (let i = 0; i < layer.units; i++) {
        let biasGradient = 0;
        
        for (let j = 0; j < n_samples; j++) {
          biasGradient += currentGradients[j][i];
        }
        
        layer.bias[i] -= learningRate * biasGradient / n_samples;
        
        for (let j = 0; j < prevOutput[0].length; j++) {
          let weightGradient = 0;
          
          for (let k = 0; k < n_samples; k++) {
            weightGradient += currentGradients[k][i] * prevOutput[k][j];
          }
          
          layer.weights[i][j] -= learningRate * weightGradient / n_samples;
        }
      }
      
      // 计算前一层的梯度（如果不是第一层）
      if (l > 0) {
        for (let i = 0; i < n_samples; i++) {
          const gradient = Array(prevOutput[i].length).fill(0);
          
          for (let j = 0; j < layer.units; j++) {
            for (let k = 0; k < prevOutput[i].length; k++) {
              gradient[k] += currentGradients[i][j] * layer.weights[j][k];
            }
          }
          
          nextGradients.push(gradient);
        }
        
        currentGradients = nextGradients;
      }
    }
  }
}

// 辅助函数：生成正态分布随机数
function randomNormal(mean = 0, stddev = 1) {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stddev + mean;
}
```

## 评估与可视化 (Evaluation and Visualization)

### 模型评估 (Model Evaluation)

```javascript
function evaluateModel(model, X_test, y_test) {
  const predictions = model.predict(X_test);
  
  // 计算准确率
  let correctCount = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === y_test[i]) {
      correctCount++;
    }
  }
  const accuracy = correctCount / predictions.length;
  
  // 计算混淆矩阵
  const uniqueLabels = [...new Set(y_test)];
  const confusionMatrix = Array(uniqueLabels.length).fill().map(() => 
    Array(uniqueLabels.length).fill(0)
  );
  
  for (let i = 0; i < predictions.length; i++) {
    const trueIndex = uniqueLabels.indexOf(y_test[i]);
    const predIndex = uniqueLabels.indexOf(predictions[i]);
    confusionMatrix[trueIndex][predIndex]++;
  }
  
  return {
    accuracy,
    confusionMatrix,
    predictions
  };
}
```

### 数据可视化 (Data Visualization)

使用Chart.js实现各种可视化：

#### 散点图 (Scatter Plot)

```javascript
function createScatterPlot(data, xFeature, yFeature, labels, containerId) {
  const ctx = document.getElementById(containerId).getContext('2d');
  
  // 准备数据点
  const datasets = [];
  const uniqueLabels = [...new Set(labels)];
  
  // 为每个类别创建一个数据集
  uniqueLabels.forEach((label, index) => {
    const points = [];
    
    for (let i = 0; i < data.length; i++) {
      if (labels[i] === label) {
        points.push({
          x: data[i][xFeature],
          y: data[i][yFeature]
        });
      }
    }
    
    datasets.push({
      label: `Class ${label}`,
      data: points,
      backgroundColor: getColor(index),
      pointRadius: 5,
      pointHoverRadius: 7
    });
  });
  
  // 创建图表
  return new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets
    },
    options: {
      responsive: true,
      title: {
        display: true,
        text: `Scatter Plot: ${xFeature} vs ${yFeature}`
      },
      scales: {
        x: {
          title: {
            display: true,
            text: xFeature
          }
        },
        y: {
          title: {
            display: true,
            text: yFeature
          }
        }
      }
    }
  });
}
```

#### 训练历史可视化 (Training History Visualization)

```javascript
function plotTrainingHistory(history, containerId) {
  const ctx = document.getElementById(containerId).getContext('2d');
  
  const epochs = Array(history.loss.length).fill().map((_, i) => i + 1);
  
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Loss',
          data: history.loss,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          yAxisID: 'y-loss'
        },
        {
          label: 'Accuracy',
          data: history.accuracy,
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          yAxisID: 'y-accuracy'
        }
      ]
    },
    options: {
      responsive: true,
      title: {
        display: true,
        text: 'Training History'
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch'
          }
        },
        'y-loss': {
          type: 'linear',
          position: 'left',
          title: {
            display: true,
            text: 'Loss'
          }
        },
        'y-accuracy': {
          type: 'linear',
          position: 'right',
          title: {
            display: true,
            text: 'Accuracy'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
}
```

#### 混淆矩阵可视化 (Confusion Matrix Visualization)

```javascript
function plotConfusionMatrix(confusionMatrix, labels, containerId) {
  const ctx = document.getElementById(containerId).getContext('2d');
  
  // 找出最大值用于颜色缩放
  const maxValue = Math.max(...confusionMatrix.flat());
  
  // 准备数据
  const data = [];
  for (let i = 0; i < confusionMatrix.length; i++) {
    for (let j = 0; j < confusionMatrix[i].length; j++) {
      data.push({
        x: j,
        y: i,
        v: confusionMatrix[i][j]
      });
    }
  }
  
  return new Chart(ctx, {
    type: 'matrix',
    data: {
      datasets: [{
        data: data,
        backgroundColor: (context) => {
          const value = context.dataset.data[context.dataIndex].v;
          const alpha = value / maxValue;
          return `rgba(54, 162, 235, ${alpha})`;
        },
        borderColor: 'white',
        borderWidth: 1,
        width: ({ chart }) => (chart.chartArea || {}).width / confusionMatrix.length - 1,
        height: ({