/**
 * 自定义Blockly块的代码生成器
 * 用于将Blockly块转换为可执行的JavaScript代码
 */

// 为每个块定义JavaScript代码生成器

// 数据处理类块的代码生成器
Blockly.JavaScript['load_dataset'] = function(block) {
  var dataset = block.getFieldValue('DATASET');
  var variable = block.getFieldValue('VAR');
  
  var code = '';
  if (dataset === 'custom_csv') {
    code = `// 加载自定义CSV数据集\n`;
    code += `async function loadCustomCSV() {\n`;
    code += `  const fileInput = document.createElement('input');\n`;
    code += `  fileInput.type = 'file';\n`;
    code += `  fileInput.accept = '.csv';\n`;
    code += `  fileInput.click();\n\n`;
    code += `  return new Promise((resolve) => {\n`;
    code += `    fileInput.onchange = async (e) => {\n`;
    code += `      const file = e.target.files[0];\n`;
    code += `      const reader = new FileReader();\n`;
    code += `      reader.onload = (event) => {\n`;
    code += `        const csvData = event.target.result;\n`;
    code += `        // 解析CSV数据\n`;
    code += `        const rows = csvData.split('\\n').map(row => row.split(','));\n`;
    code += `        const headers = rows[0];\n`;
    code += `        const data = rows.slice(1).filter(row => row.length === headers.length);\n`;
    code += `        // 转换为数值型数据\n`;
    code += `        const numericData = data.map(row => row.map(val => parseFloat(val) || 0));\n`;
    code += `        console.log('已加载自定义CSV数据集，形状：[' + numericData.length + ', ' + headers.length + ']');\n`;
    code += `        resolve({\n`;
    code += `          data: numericData,\n`;
    code += `          headers: headers,\n`;
    code += `          features: numericData.map(row => row.slice(0, -1)),\n`;
    code += `          labels: numericData.map(row => row[row.length - 1])\n`;
    code += `        });\n`;
    code += `      };\n`;
    code += `      reader.readAsText(file);\n`;
    code += `    };\n`;
    code += `  });\n`;
    code += `}\n\n`;
    code += `let ${variable} = await loadCustomCSV();\n`;
  } else {
    code = `// 加载预定义数据集: ${dataset}\n`;
    code += `async function loadDataset() {\n`;
    code += `  console.log('正在加载${dataset}数据集...');\n`;
    
    if (dataset === 'iris') {
      code += `  const irisData = await fetch('https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/iris-fitDataset/data/iris.csv')\n`;
      code += `    .then(response => response.text())\n`;
      code += `    .then(text => {\n`;
      code += `      const rows = text.split('\\n').filter(row => row.length > 0);\n`;
      code += `      const headers = rows[0].split(',');\n`;
      code += `      const data = rows.slice(1).map(row => {\n`;
      code += `        const values = row.split(',');\n`;
      code += `        return {\n`;
      code += `          features: values.slice(0, 4).map(v => parseFloat(v)),\n`;
      code += `          label: values[4]\n`;
      code += `        };\n`;
      code += `      });\n`;
      code += `      // 将标签转换为数字\n`;
      code += `      const labelMap = {};\n`;
      code += `      data.forEach(item => {\n`;
      code += `        if (!labelMap[item.label]) {\n`;
      code += `          labelMap[item.label] = Object.keys(labelMap).length;\n`;
      code += `        }\n`;
      code += `      });\n`;
      code += `      data.forEach(item => {\n`;
      code += `        item.labelIndex = labelMap[item.label];\n`;
      code += `      });\n`;
      code += `      console.log('已加载Iris数据集，共' + data.length + '条记录');\n`;
      code += `      return {\n`;
      code += `        data: data,\n`;
      code += `        features: data.map(d => d.features),\n`;
      code += `        labels: data.map(d => d.labelIndex),\n`;
      code += `        labelMap: labelMap\n`;
      code += `      };\n`;
      code += `    });\n`;
      code += `  return irisData;\n`;
    } else if (dataset === 'boston') {
      code += `  // 波士顿房价数据集\n`;
      code += `  const bostonData = await fetch('https://storage.googleapis.com/tfjs-examples/boston-housing/data/boston-housing-train.csv')\n`;
      code += `    .then(response => response.text())\n`;
      code += `    .then(text => {\n`;
      code += `      const rows = text.split('\\n').filter(row => row.length > 0);\n`;
      code += `      const data = rows.map(row => {\n`;
      code += `        const values = row.split(',').map(v => parseFloat(v));\n`;
      code += `        return {\n`;
      code += `          features: values.slice(0, values.length - 1),\n`;
      code += `          label: values[values.length - 1]\n`;
      code += `        };\n`;
      code += `      });\n`;
      code += `      console.log('已加载Boston房价数据集，共' + data.length + '条记录');\n`;
      code += `      return {\n`;
      code += `        data: data,\n`;
      code += `        features: data.map(d => d.features),\n`;
      code += `        labels: data.map(d => d.label)\n`;
      code += `      };\n`;
      code += `    });\n`;
      code += `  return bostonData;\n`;
    } else if (dataset === 'mnist' || dataset === 'fashion_mnist') {
      const datasetName = dataset === 'mnist' ? 'MNIST手写数字' : 'Fashion MNIST时尚物品';
      code += `  // 加载${datasetName}数据集\n`;
      code += `  console.log('${datasetName}数据集较大，加载可能需要一些时间...');\n`;
      code += `  const data = {};\n`;
      code += `  await tf.data.${dataset}()\n`;
      code += `    .take(1000) // 限制样本数量，避免浏览器内存不足\n`;
      code += `    .forEachAsync(e => {\n`;
      code += `      if (!data.features) {\n`;
      code += `        data.features = [];\n`;
      code += `        data.labels = [];\n`;
      code += `      }\n`;
      code += `      // 将图像数据展平为一维数组\n`;
      code += `      const flattenedX = e.xs.reshape([e.xs.shape[1] * e.xs.shape[2]]).arraySync();\n`;
      code += `      data.features.push(flattenedX);\n`;
      code += `      data.labels.push(e.ys.argMax(-1).arraySync()[0]);\n`;
      code += `    });\n`;
      code += `  console.log('已加载${datasetName}数据集，共' + data.features.length + '条记录');\n`;
      code += `  return data;\n`;
    } else if (dataset === 'cifar10') {
      code += `  // 加载CIFAR10数据集\n`;
      code += `  console.log('CIFAR10数据集较大，加载可能需要一些时间...');\n`;
      code += `  const data = {};\n`;
      code += `  data.features = [];\n`;
      code += `  data.labels = [];\n`;
      code += `  const response = await fetch('https://storage.googleapis.com/tfjs-examples/cifar-10/data/cifar10.json');\n`;
      code += `  const cifar = await response.json();\n`;
      code += `  // 限制样本数量，避免浏览器内存不足\n`;
      code += `  const samples = cifar.slice(0, 1000);\n`;
      code += `  for (const sample of samples) {\n`;
      code += `    data.features.push(sample.xs);\n`;
      code += `    data.labels.push(sample.ys);\n`;
      code += `  }\n`;
      code += `  console.log('已加载CIFAR10数据集，共' + data.features.length + '条记录');\n`;
      code += `  return data;\n`;
    }
    
    code += `}\n\n`;
    code += `let ${variable} = await loadDataset();\n`;
  }
  
  return code;
};

Blockly.JavaScript['split_data'] = function(block) {
  var datasetVar = Blockly.JavaScript.valueToCode(block, 'DATASET', Blockly.JavaScript.ORDER_ATOMIC) || 'dataset';
  var testSize = block.getFieldValue('TEST_SIZE');
  var randomState = block.getFieldValue('RANDOM_STATE');
  var outputVars = block.getFieldValue('VAR');
  
  // 解析输出变量名
  var varNames = outputVars.split(',').map(name => name.trim());
  if (varNames.length !== 4) {
    varNames = ['X_train', 'X_test', 'y_train', 'y_test'];
  }
  
  var code = `// 分割数据集为训练集和测试集\n`;
  code += `function splitData(data, testSize = ${testSize}, seed = ${randomState}) {\n`;
  code += `  // 设置随机种子\n`;
  code += `  const shuffledIndices = Array.from(Array(data.features.length).keys());\n`;
  code += `  // 简单的伪随机洗牌\n`;
  code += `  for (let i = shuffledIndices.length - 1; i > 0; i--) {\n`;
  code += `    const j = Math.floor((Math.random() * seed) % (i + 1));\n`;
  code += `    [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];\n`;
  code += `  }\n\n`;
  code += `  const numSamples = data.features.length;\n`;
  code += `  const numTest = Math.round(numSamples * testSize);\n`;
  code += `  const numTrain = numSamples - numTest;\n\n`;
  code += `  // 分割数据\n`;
  code += `  const ${varNames[0]} = [];\n`;
  code += `  const ${varNames[1]} = [];\n`;
  code += `  const ${varNames[2]} = [];\n`;
  code += `  const ${varNames[3]} = [];\n\n`;
  code += `  for (let i = 0; i < numSamples; i++) {\n`;
  code += `    const idx = shuffledIndices[i];\n`;
  code += `    if (i < numTrain) {\n`;
  code += `      ${varNames[0]}.push(data.features[idx]);\n`;
  code += `      ${varNames[2]}.push(data.labels[idx]);\n`;
  code += `    } else {\n`;
  code += `      ${varNames[1]}.push(data.features[idx]);\n`;
  code += `      ${varNames[3]}.push(data.labels[idx]);\n`;
  code += `    }\n`;
  code += `  }\n\n`;
  code += `  console.log('数据集已分割：训练集大小 = ' + ${varNames[0]}.length + ', 测试集大小 = ' + ${varNames[1]}.length);\n`;
  code += `  return { ${varNames[0]}, ${varNames[1]}, ${varNames[2]}, ${varNames[3]} };\n`;
  code += `}\n\n`;
  code += `const { ${varNames[0]}, ${varNames[1]}, ${varNames[2]}, ${varNames[3]} } = splitData(${datasetVar});\n`;
  
  return code;
};

Blockly.JavaScript['normalize_data'] = function(block) {
  var dataVar = Blockly.JavaScript.valueToCode(block, 'DATA', Blockly.JavaScript.ORDER_ATOMIC) || 'data';
  var method = block.getFieldValue('METHOD');
  var outputVar = block.getFieldValue('VAR');
  
  var code = `// 归一化数据\n`;
  code += `function normalizeData(data, method = '${method}') {\n`;
  
  if (method === 'standard') {
    code += `  // 标准化 (StandardScaler): (x - mean) / std\n`;
    code += `  // 使用类似PyTorch的操作方式\n`;
    code += `  // 计算每列的均值\n`;
    code += `  const mean = data[0].map((_, colIndex) => {\n`;
    code += `    return data.reduce((sum, row) => sum + row[colIndex], 0) / data.length;\n`;
    code += `  });\n`;
    code += `  // 计算每列的标准差\n`;
    code += `  const std = data[0].map((_, colIndex) => {\n`;
    code += `    const colMean = mean[colIndex];\n`;
    code += `    const sumSquares = data.reduce((sum, row) => sum + Math.pow(row[colIndex] - colMean, 2), 0);\n`;
    code += `    return Math.sqrt(sumSquares / data.length);\n`;
    code += `  });\n`;
    code += `  // 应用标准化\n`;
    code += `  const normalized = data.map(row => {\n`;
    code += `    return row.map((val, colIndex) => (val - mean[colIndex]) / (std[colIndex] || 1));\n`;
    code += `  });\n`;
    code += `  console.log('已使用标准化(StandardScaler)方法归一化数据');\n`;
    code += `  return normalized;\n`;
  } else if (method === 'minmax') {
    code += `  // 最小最大缩放 (MinMaxScaler): (x - min) / (max - min)\n`;
    code += `  const dataTensor = tf.tensor2d(data);\n`;
    code += `  const min = dataTensor.min(0);\n`;
    code += `  const max = dataTensor.max(0);\n`;
    code += `  const normalized = dataTensor.sub(min).div(max.sub(min));\n`;
    code += `  console.log('已使用最小最大缩放(MinMaxScaler)方法归一化数据');\n`;
    code += `  return normalized.arraySync();\n`;
  } else if (method === 'robust') {
    code += `  // 鲁棒缩放 (RobustScaler): (x - median) / (q3 - q1)\n`;
    code += `  const dataTensor = tf.tensor2d(data);\n`;
    code += `  // 计算中位数（近似）\n`;
    code += `  const sortedVals = Array.from(Array(data[0].length).keys()).map(j => {\n`;
    code += `    return data.map(row => row[j]).sort((a, b) => a - b);\n`;
    code += `  });\n`;
    code += `  const medians = sortedVals.map(col => {\n`;
    code += `    const mid = Math.floor(col.length / 2);\n`;
    code += `    return col.length % 2 === 0 ? (col[mid - 1] + col[mid]) / 2 : col[mid];\n`;
    code += `  });\n`;
    code += `  // 计算四分位数\n`;
    code += `  const q1 = sortedVals.map(col => {\n`;
    code += `    const idx = Math.floor(col.length / 4);\n`;
    code += `    return col[idx];\n`;
    code += `  });\n`;
    code += `  const q3 = sortedVals.map(col => {\n`;
    code += `    const idx = Math.floor(3 * col.length / 4);\n`;
    code += `    return col[idx];\n`;
    code += `  });\n`;
    code += `  // 应用鲁棒缩放\n`;
    code += `  const normalized = data.map(row => {\n`;
    code += `    return row.map((val, j) => {\n`;
    code += `      const iqr = q3[j] - q1[j];\n`;
    code += `      return iqr === 0 ? 0 : (val - medians[j]) / iqr;\n`;
    code += `    });\n`;
    code += `  });\n`;
    code += `  console.log('已使用鲁棒缩放(RobustScaler)方法归一化数据');\n`;
    code += `  return normalized;\n`;
  }
  
  code += `}\n\n`;
  code += `const ${outputVar} = normalizeData(${dataVar});\n`;
  
  return code;
};

// 模型构建类块的代码生成器
Blockly.JavaScript['create_linear_model'] = function(block) {
  var modelType = block.getFieldValue('MODEL_TYPE');
  var modelVar = block.getFieldValue('VAR');
  
  var code = `// 创建${modelType}模型\n`;
  code += `function createLinearModel(modelType = '${modelType}') {\n`;
  
  if (modelType === 'linear_regression') {
    code += `  // 线性回归模型 (PyTorch风格)\n`;
    code += `  // 注意：这里使用的是PyTorch风格的API，但在前端环境中是通过ONNX.js实现\n`;
    code += `  const model = {\n`;
    code += `    type: 'linear_regression',\n`;
    code += `    weights: null,  // 训练后填充\n`;
    code += `    bias: null,     // 训练后填充\n`;
    code += `    config: {\n`;
    code += `      learning_rate: 0.01,\n`;
    code += `      optimizer: 'adam',\n`;
    code += `      loss: 'mse'\n`;
    code += `    },\n`;
    code += `    forward: function(X) {\n`;
    code += `      // 线性回归前向传播\n`;
    code += `      if (!this.weights) {\n`;
    code += `        // 初始化权重\n`;
    code += `        const inputDim = X[0].length;\n`;
    code += `        this.weights = Array(inputDim).fill(0).map(() => Math.random() * 0.1);\n`;
    code += `        this.bias = Math.random() * 0.1;\n`;
    code += `      }\n`;
    code += `      return X.map(x => {\n`;
    code += `        let sum = this.bias;\n`;
    code += `        for (let i = 0; i < x.length; i++) {\n`;
    code += `          sum += x[i] * this.weights[i];\n`;
    code += `        }\n`;
    code += `        return sum;\n`;
    code += `      });\n`;
    code += `    }\n`;
    code += `  };\n`;
    code += `  console.log('已创建线性回归模型 (PyTorch风格)');\n`;
  } else if (modelType === 'logistic_regression') {
    code += `  // 逻辑回归模型\n`;
    code += `  const model = tf.sequential();\n`;
    code += `  model.add(tf.layers.dense({units: 1, inputShape: [null], activation: 'sigmoid'}));\n`;
    code += `  model.compile({\n`;
    code += `    optimizer: tf.train.adam(0.01),\n`;
    code += `    loss: 'binaryCrossentropy',\n`;
    code += `    metrics: ['accuracy']\n`;
    code += `  });\n`;
    code += `  console.log('已创建逻辑回归模型');\n`;
  } else if (modelType === 'ridge') {
    code += `  // 岭回归模型 (L2正则化)\n`;
    code += `  const model = tf.sequential();\n`;
    code += `  model.add(tf.layers.dense({\n`;
    code += `    units: 1, \n`;
    code += `    inputShape: [null], \n`;
    code += `    activation: 'linear',\n`;
    code += `    kernelRegularizer: tf.regularizers.l2({l2: 0.01})\n`;
    code += `  }));\n`;
    code += `  model.compile({\n`;
    code += `    optimizer: tf.train.adam(0.01),\n`;
    code += `    loss: 'meanSquaredError',\n`;
    code += `    metrics: ['mse']\n`;
    code += `  });\n`;
    code += `  console.log('已创建岭回归模型');\n`;
  } else if (modelType === 'lasso') {
    code += `  // Lasso回归模型 (L1正则化)\n`;
    code += `  const model = tf.sequential();\n`;
    code += `  model.add(tf.layers.dense({\n`;
    code += `    units: 1, \n`;
    code += `    inputShape: [null], \n`;
    code += `    activation: 'linear',\n`;
    code += `    kernelRegularizer: tf.regularizers.l1({l1: 0.01})\n`;
    code += `  }));\n`;
    code += `  model.compile({\n`;
    code += `    optimizer: tf.train.adam(0.01),\n`;
    code += `    loss: 'meanSquaredError',\n`;
    code += `    metrics: ['mse']\n`;
    code += `  });\n`;
    code += `  console.log('已创建Lasso回归模型');\n`;
  }
  
  code += `  return model;\n`;
  code += `}\n\n`;
  code += `const ${modelVar} = createLinearModel();\n`;
  
  return code;
};

Blockly.JavaScript['create_neural_network'] = function(block) {
  var inputDim = block.getFieldValue('INPUT_DIM');
  var modelVar = block.getFieldValue('VAR');
  
  var code = `// 创建神经网络模型\n`;
  code += `function createNeuralNetwork(inputDim = ${inputDim}) {\n`;
  code += `  const model = tf.sequential();\n`;
  code += `  // 初始化模型，但不添加任何层\n`;
  code += `  console.log('已创建神经网络模型，输入维度: ' + inputDim);\n`;
  code += `  return {\n`;
  code += `    model: model,\n`;
  code += `    inputDim: inputDim,\n`;
  code += `    isFirstLayer: true\n`;
  code += `  };\n`;
  code += `}\n\n`;
  code += `const ${modelVar} = createNeuralNetwork();\n`;
  
  return code;
};

Blockly.JavaScript['add_layer'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var layerType = block.getFieldValue('LAYER_TYPE');
  var units = block.getFieldValue('UNITS');
  var activation = block.getFieldValue('ACTIVATION');
  
  var code = `// 添加${layerType}层\n`;
  code += `function addLayer(modelObj, layerType = '${layerType}', units = ${units}, activation = '${activation}') {\n`;
  code += `  if (!modelObj || !modelObj.model) {\n`;
  code += `    console.error('错误：无效的模型对象');\n`;
  code += `    return modelObj;\n`;
  code += `  }\n\n`;
  
  if (layerType === 'dense') {
    code += `  // 添加全连接层\n`;
    code += `  if (modelObj.isFirstLayer) {\n`;
    code += `    modelObj.model.add(tf.layers.dense({\n`;
    code += `      units: units,\n`;
    code += `      inputShape: [modelObj.inputDim],\n`;
    code += `      activation: activation === 'none' ? null : activation\n`;
    code += `    }));\n`;
    code += `    modelObj.isFirstLayer = false;\n`;
    code += `  } else {\n`;
    code += `    modelObj.model.add(tf.layers.dense({\n`;
    code += `      units: units,\n`;
    code += `      activation: activation === 'none' ? null : activation\n`;
    code += `    }));\n`;
    code += `  }\n`;
    code += `  console.log('已添加全连接层，神经元数量: ' + units + ', 激活函数: ' + (activation === 'none' ? '无' : activation));\n`;
  } else if (layerType === 'dropout') {
    code += `  // 添加Dropout层\n`;
    code += `  const rate = units / 100; // 将单位转换为比例 (0-1)\n`;
    code += `  modelObj.model.add(tf.layers.dropout({rate: rate}));\n`;
    code += `  console.log('已添加Dropout层，丢弃率: ' + rate);\n`;
  } else if (layerType === 'conv2d') {
    code += `  // 添加二维卷积层\n`;
    code += `  if (modelObj.isFirstLayer) {\n`;
    code += `    modelObj.model.add(tf.layers.conv2d({\n`;
    code += `      filters: units,\n`;
    code += `      kernelSize: 3,\n`;
    code += `      padding: 'same',\n`;
    code += `      activation: activation === 'none' ? null : activation,\n`;
    code += `      inputShape: [modelObj.inputDim, modelObj.inputDim, 3] // 假设输入是方形图像，3通道\n`;
    code += `    }));\n`;
    code += `    modelObj.isFirstLayer = false;\n`;
    code += `  } else {\n`;
    code += `    modelObj.model.add(tf.layers.conv2d({\n`;
    code += `      filters: units,\n`;
    code += `      kernelSize: 3,\n`;
    code += `      padding: 'same',\n`;
    code += `      activation: activation === 'none' ? null : activation\n`;
    code += `    }));\n`;
    code += `  }\n`;
    code += `  console.log('已添加二维卷积层，过滤器数量: ' + units + ', 激活函数: ' + (activation === 'none' ? '无' : activation));\n`;
  } else if (layerType === 'maxpool2d') {
    code += `  // 添加二维最大池化层\n`;
    code += `  modelObj.model.add(tf.layers.maxPooling2d({\n`;
    code += `    poolSize: [2, 2],\n`;
    code += `    strides: [2, 2]\n`;
    code += `  }));\n`;
    code += `  console.log('已添加二维最大池化层，池化大小: 2x2');\n`;
  }
  
  code += `  return modelObj;\n`;
  code += `}\n\n`;
  code += `${modelVar} = addLayer(${modelVar});\n`;
  
  return code;
};

// 训练与评估类块的代码生成器
Blockly.JavaScript['train_model'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var xTrainVar = Blockly.JavaScript.valueToCode(block, 'X_TRAIN', Blockly.JavaScript.ORDER_ATOMIC) || 'X_train';
  var yTrainVar = Blockly.JavaScript.valueToCode(block, 'Y_TRAIN', Blockly.JavaScript.ORDER_ATOMIC) || 'y_train';
  var batchSize = block.getFieldValue('BATCH_SIZE');
  var epochs = block.getFieldValue('EPOCHS');
  var historyVar = block.getFieldValue('HISTORY_VAR');
  
  var code = `// 训练模型\n`;
  code += `async function trainModel(model, xTrain, yTrain, batchSize = ${batchSize}, epochs = ${epochs}) {\n`;
  code += `  // 检查模型对象类型\n`;
  code += `  const actualModel = model.model ? model.model : model;\n\n`;
  code += `  // 准备训练数据\n`;
  code += `  const xs = tf.tensor2d(xTrain);\n`;
  code += `  // 检查标签是否需要进行one-hot编码\