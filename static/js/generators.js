/**
 * Custom Blockly block code generators
 * Used to convert Blockly blocks to executable JavaScript code
 */

// Define JavaScript code generators for each block

// Data processing block code generators
Blockly.JavaScript['load_dataset'] = function(block) {
  var dataset = block.getFieldValue('DATASET');
  var variable = block.getFieldValue('VAR');
  
  var code = '';
  if (dataset === 'custom_csv') {
    code = `// Loading custom CSV dataset\n`;
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
    code += `        // Parse CSV data\n`;
    code += `        const rows = csvData.split('\\n').map(row => row.split(','));\n`;
    code += `        const headers = rows[0];\n`;
    code += `        const data = rows.slice(1).filter(row => row.length === headers.length);\n`;
    code += `        // Convert to numeric data\n`;
    code += `        const numericData = data.map(row => row.map(val => parseFloat(val) || 0));\n`;
    code += `        console.log('Custom CSV dataset loaded, shape: [' + numericData.length + ', ' + headers.length + ']');\n`;
    code += `        const result = {\n`;
    code += `          data: numericData,\n`;
    code += `          headers: headers,\n`;
    code += `          features: numericData.map(row => row.slice(0, -1)),\n`;
    code += `          labels: numericData.map(row => row[row.length - 1])\n`;
    code += `        };\n`;
    code += `        // Set the dataset globally for visualization\n`;
    code += `        setCurrentDataset(result);\n`;
    code += `        resolve(result);\n`;
    code += `      };\n`;
    code += `      reader.readAsText(file);\n`;
    code += `    };\n`;
    code += `  });\n`;
    code += `}\n\n`;
    code += `let ${variable} = await loadCustomCSV();\n`;
  } else {
    code = `// Loading predefined dataset: ${dataset}\n`;
    code += `async function loadDataset() {\n`;
    code += `  console.log('Loading ${dataset} dataset...');\n`;
    
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
      code += `      // Convert labels to numbers\n`;
      code += `      const labelMap = {};\n`;
      code += `      data.forEach(item => {\n`;
      code += `        if (!labelMap[item.label]) {\n`;
      code += `          labelMap[item.label] = Object.keys(labelMap).length;\n`;
      code += `        }\n`;
      code += `      });\n`;
      code += `      data.forEach(item => {\n`;
      code += `        item.labelIndex = labelMap[item.label];\n`;
      code += `      });\n`;
      code += `      console.log('Iris dataset loaded, ' + data.length + ' records');\n`;
      code += `      const result = {\n`;
      code += `        data: data,\n`;
      code += `        features: data.map(d => d.features),\n`;
      code += `        labels: data.map(d => d.labelIndex),\n`;
      code += `        labelMap: labelMap\n`;
      code += `      };\n`;
      code += `      // Set the dataset globally for visualization\n`;
      code += `      setCurrentDataset(result);\n`;
      code += `      return result;\n`;
      code += `    });\n`;
      code += `  return irisData;\n`;
    } else if (dataset === 'boston') {
      code += `  // Boston Housing dataset\n`;
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
      code += `      console.log('Boston Housing dataset loaded, ' + data.length + ' records');\n`;
      code += `      const result = {\n`;
      code += `        data: data,\n`;
      code += `        features: data.map(d => d.features),\n`;
      code += `        labels: data.map(d => d.label)\n`;
      code += `      };\n`;
      code += `      // Set the dataset globally for visualization\n`;
      code += `      setCurrentDataset(result);\n`;
      code += `      return result;\n`;
      code += `    });\n`;
      code += `  return bostonData;\n`;
    } else if (dataset === 'mnist' || dataset === 'fashion_mnist') {
      const datasetName = dataset === 'mnist' ? 'MNIST handwritten digits' : 'Fashion MNIST items';
      code += `  // Loading ${datasetName} dataset\n`;
      code += `  console.log('${datasetName} dataset is large, loading may take some time...');\n`;
      code += `  const data = {};\n`;
      code += `  await tf.data.${dataset}()\n`;
      code += `    .take(1000) // Limit sample size to avoid browser memory issues\n`;
      code += `    .forEachAsync(e => {\n`;
      code += `      if (!data.features) {\n`;
      code += `        data.features = [];\n`;
      code += `        data.labels = [];\n`;
      code += `      }\n`;
      code += `      // Flatten image data to 1D array\n`;
      code += `      const flattenedX = e.xs.reshape([e.xs.shape[1] * e.xs.shape[2]]).arraySync();\n`;
      code += `      data.features.push(flattenedX);\n`;
      code += `      data.labels.push(e.ys.argMax(-1).arraySync()[0]);\n`;
      code += `    });\n`;
      code += `  console.log('${datasetName} dataset loaded, ' + data.features.length + ' records');\n`;
      code += `  // Set the dataset globally for visualization\n`;
      code += `  setCurrentDataset(data);\n`;
      code += `  return data;\n`;
    } else if (dataset === 'cifar10') {
      code += `  // Loading CIFAR10 dataset\n`;
      code += `  console.log('CIFAR10 dataset is large, loading may take some time...');\n`;
      code += `  const data = {};\n`;
      code += `  data.features = [];\n`;
      code += `  data.labels = [];\n`;
      code += `  const response = await fetch('https://storage.googleapis.com/tfjs-examples/cifar-10/data/cifar10.json');\n`;
      code += `  const cifar = await response.json();\n`;
      code += `  // Limit sample size to avoid browser memory issues\n`;
      code += `  const samples = cifar.slice(0, 1000);\n`;
      code += `  for (const sample of samples) {\n`;
      code += `    data.features.push(sample.xs);\n`;
      code += `    data.labels.push(sample.ys);\n`;
      code += `  }\n`;
      code += `  console.log('CIFAR10 dataset loaded, ' + data.features.length + ' records');\n`;
      code += `  // Set the dataset globally for visualization\n`;
      code += `  setCurrentDataset(data);\n`;
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
  
  // Parse output variable names
  var varNames = outputVars.split(',').map(name => name.trim());
  if (varNames.length !== 4) {
    varNames = ['X_train', 'X_test', 'y_train', 'y_test'];
  }
  
  var code = `// Split dataset into training and test sets\n`;
  code += `function splitData(data, testSize = ${testSize}, seed = ${randomState}) {\n`;
  code += `  // Set random seed\n`;
  code += `  const shuffledIndices = Array.from(Array(data.features.length).keys());\n`;
  code += `  // Simple pseudo-random shuffle\n`;
  code += `  for (let i = shuffledIndices.length - 1; i > 0; i--) {\n`;
  code += `    const j = Math.floor((Math.random() * seed) % (i + 1));\n`;
  code += `    [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];\n`;
  code += `  }\n\n`;
  code += `  const numSamples = data.features.length;\n`;
  code += `  const numTest = Math.round(numSamples * testSize);\n`;
  code += `  const numTrain = numSamples - numTest;\n\n`;
  code += `  // Split data\n`;
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
  code += `  console.log('Dataset split: Training set size = ' + ${varNames[0]}.length + ', Test set size = ' + ${varNames[1]}.length);\n`;
  code += `  return { ${varNames[0]}, ${varNames[1]}, ${varNames[2]}, ${varNames[3]} };\n`;
  code += `}\n\n`;
  code += `const { ${varNames[0]}, ${varNames[1]}, ${varNames[2]}, ${varNames[3]} } = splitData(${datasetVar});\n`;
  
  return code;
};

Blockly.JavaScript['normalize_data'] = function(block) {
  var dataVar = Blockly.JavaScript.valueToCode(block, 'DATA', Blockly.JavaScript.ORDER_ATOMIC) || 'data';
  var method = block.getFieldValue('METHOD');
  var outputVar = block.getFieldValue('VAR');
  
  var code = `// Normalize data\n`;
  code += `function normalizeData(data, method = '${method}') {\n`;
  
  if (method === 'standard') {
    code += `  // Standardization (StandardScaler): (x - mean) / std\n`;
    code += `  // Using PyTorch-like operations\n`;
    code += `  // Calculate mean for each column\n`;
    code += `  const mean = data[0].map((_, colIndex) => {\n`;
    code += `    return data.reduce((sum, row) => sum + row[colIndex], 0) / data.length;\n`;
    code += `  });\n`;
    code += `  // Calculate standard deviation for each column\n`;
    code += `  const std = data[0].map((_, colIndex) => {\n`;
    code += `    const colMean = mean[colIndex];\n`;
    code += `    const sumSquares = data.reduce((sum, row) => sum + Math.pow(row[colIndex] - colMean, 2), 0);\n`;
    code += `    return Math.sqrt(sumSquares / data.length);\n`;
    code += `  });\n`;
    code += `  // Apply standardization\n`;
    code += `  const normalized = data.map(row => {\n`;
    code += `    return row.map((val, colIndex) => (val - mean[colIndex]) / (std[colIndex] || 1));\n`;
    code += `  });\n`;
    code += `  console.log('Data normalized using Standardization (StandardScaler) method');\n`;
    code += `  return normalized;\n`;
  } else if (method === 'minmax') {
    code += `  // Min-Max Scaling (MinMaxScaler): (x - min) / (max - min)\n`;
    code += `  const dataTensor = tf.tensor2d(data);\n`;
    code += `  const min = dataTensor.min(0);\n`;
    code += `  const max = dataTensor.max(0);\n`;
    code += `  const normalized = dataTensor.sub(min).div(max.sub(min));\n`;
    code += `  console.log('Data normalized using Min-Max Scaling (MinMaxScaler) method');\n`;
    code += `  return normalized.arraySync();\n`;
  } else if (method === 'robust') {
    code += `  // Robust Scaling (RobustScaler): (x - median) / (q3 - q1)\n`;
    code += `  const dataTensor = tf.tensor2d(data);\n`;
    // Calculate median (approximate)
    code += `  const sortedVals = Array.from(Array(data[0].length).keys()).map(j => {\n`;
    code += `    return data.map(row => row[j]).sort((a, b) => a - b);\n`;
    code += `  });\n`;
    code += `  const medians = sortedVals.map(col => {\n`;
    code += `    const mid = Math.floor(col.length / 2);\n`;
    code += `    return col.length % 2 === 0 ? (col[mid - 1] + col[mid]) / 2 : col[mid];\n`;
    code += `  });\n`;
    // Calculate quartiles
    code += `  const q1 = sortedVals.map(col => {\n`;
    code += `    const idx = Math.floor(col.length / 4);\n`;
    code += `    return col[idx];\n`;
    code += `  });\n`;
    code += `  const q3 = sortedVals.map(col => {\n`;
    code += `    const idx = Math.floor(3 * col.length / 4);\n`;
    code += `    return col[idx];\n`;
    code += `  });\n`;
    // Apply robust scaling
    code += `  const normalized = data.map(row => {\n`;
    code += `    return row.map((val, j) => {\n`;
    code += `      const iqr = q3[j] - q1[j];\n`;
    code += `      return iqr === 0 ? 0 : (val - medians[j]) / iqr;\n`;
    code += `    });\n`;
    code += `  });\n`;
    code += `  console.log('Data normalized using Robust Scaling (RobustScaler) method');\n`;
    code += `  return normalized;\n`;
  }
  
  code += `}\n\n`;
  code += `const ${outputVar} = normalizeData(${dataVar});\n`;
  
  return code;
};

// Model building block code generators
Blockly.JavaScript['create_linear_model'] = function(block) {
  var modelType = block.getFieldValue('MODEL_TYPE');
  var modelVar = block.getFieldValue('VAR');
  
  var code = `// Create ${modelType} model\n`;
  code += `function createLinearModel(modelType = '${modelType}') {\n`;
  
  if (modelType === 'linear_regression') {
    code += `  // Linear regression model (PyTorch style)\n`;
    code += `  // Note: This uses PyTorch-style API, but is implemented through ONNX.js in the frontend\n`;
    code += `  const model = {\n`;
    code += `    type: 'linear_regression',\n`;
    code += `    weights: null,  // Filled during training\n`;
    code += `    bias: null,     // Filled during training\n`;
    code += `    config: {\n`;
    code += `      learning_rate: 0.01,\n`;
    code += `      optimizer: 'adam',\n`;
    code += `      loss: 'mse'\n`;
    code += `    },\n`;
    code += `    forward: function(X) {\n`;
    code += `      // Linear regression forward pass\n`;
    code += `      if (!this.weights) {\n`;
    code += `        // Initialize weights\n`;
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
    code += `  console.log('Linear regression model created (PyTorch style)');\n`;
  } else if (modelType === 'logistic_regression') {
    code += `  // Logistic regression model\n`;
    code += `  const model = tf.sequential();\n`;
    code += `  model.add(tf.layers.dense({units: 1, inputShape: [null], activation: 'sigmoid'}));\n`;
    code += `  model.compile({\n`;
    code += `    optimizer: tf.train.adam(0.01),\n`;
    code += `    loss: 'binaryCrossentropy',\n`;
    code += `    metrics: ['accuracy']\n`;
    code += `  });\n`;
    code += `  console.log('Logistic regression model created');\n`;
  } else if (modelType === 'ridge') {
    code += `  // Ridge regression model (L2 regularization)\n`;
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
    code += `  console.log('Ridge regression model created');\n`;
  } else if (modelType === 'lasso') {
    code += `  // Lasso regression model (L1 regularization)\n`;
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
    code += `  console.log('Lasso regression model created');\n`;
  }
  
  code += `  return model;\n`;
  code += `}\n\n`;
  code += `const ${modelVar} = createLinearModel();\n`;
  
  return code;
};

Blockly.JavaScript['create_neural_network'] = function(block) {
  var inputDim = block.getFieldValue('INPUT_DIM');
  var modelVar = block.getFieldValue('VAR');
  
  var code = `// Create neural network model\n`;
  code += `function createNeuralNetwork(inputDim = ${inputDim}) {\n`;
  code += `  const model = tf.sequential();\n`;
  // Initialize model, but don't add any layers yet
  code += `  console.log('Neural network model created, input dimension: ' + inputDim);\n`;
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
  
  var code = `// Add ${layerType} layer\n`;
  code += `function addLayer(modelObj, layerType = '${layerType}', units = ${units}, activation = '${activation}') {\n`;
  code += `  if (!modelObj || !modelObj.model) {\n`;
  code += `    console.error('Error: Invalid model object');\n`;
  code += `    return modelObj;\n`;
  code += `  }\n\n`;
  
  if (layerType === 'dense') {
    code += `  // Add dense layer\n`;
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
    code += `  console.log('Dense layer added, neuron count: ' + units + ', Activation function: ' + (activation === 'none' ? 'None' : activation));\n`;
  } else if (layerType === 'dropout') {
    code += `  // Add Dropout layer\n`;
    code += `  const rate = units / 100; // Convert unit to proportion (0-1)\n`;
    code += `  modelObj.model.add(tf.layers.dropout({rate: rate}));\n`;
    code += `  console.log('Dropout layer added, dropout rate: ' + rate);\n`;
  } else if (layerType === 'conv2d') {
    code += `  // Add 2D convolution layer\n`;
    code += `  if (modelObj.isFirstLayer) {\n`;
    code += `    modelObj.model.add(tf.layers.conv2d({\n`;
    code += `      filters: units,\n`;
    code += `      kernelSize: 3,\n`;
    code += `      padding: 'same',\n`;
    code += `      activation: activation === 'none' ? null : activation,\n`;
    code += `      inputShape: [modelObj.inputDim, modelObj.inputDim, 3] // Assume input is square image, 3 channels\n`;
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
    code += `  console.log('2D convolution layer added, filter count: ' + units + ', Activation function: ' + (activation === 'none' ? 'None' : activation));\n`;
  } else if (layerType === 'maxpool2d') {
    code += `  // Add 2D max pooling layer\n`;
    code += `  modelObj.model.add(tf.layers.maxPooling2d({\n`;
    code += `    poolSize: [2, 2],\n`;
    code += `    strides: [2, 2]\n`;
    code += `  }));\n`;
    code += `  console.log('2D max pooling layer added, pooling size: 2x2');\n`;
  }
  
  code += `  return modelObj;\n`;
  code += `}\n\n`;
  code += `${modelVar} = addLayer(${modelVar});\n`;
  
  return code;
};

// Training and evaluation block code generators
Blockly.JavaScript['train_model'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var xTrainVar = Blockly.JavaScript.valueToCode(block, 'X_TRAIN', Blockly.JavaScript.ORDER_ATOMIC) || 'X_train';
  var yTrainVar = Blockly.JavaScript.valueToCode(block, 'Y_TRAIN', Blockly.JavaScript.ORDER_ATOMIC) || 'y_train';
  var batchSize = block.getFieldValue('BATCH_SIZE');
  var epochs = block.getFieldValue('EPOCHS');
  var historyVar = block.getFieldValue('HISTORY_VAR');
  
  var code = `// Train model\n`;
  code += `async function trainModel(model, xTrain, yTrain, batchSize = ${batchSize}, epochs = ${epochs}) {\n`;
  code += `  // Check model object type\n`;
  code += `  const actualModel = model.model ? model.model : model;\n\n`;
  code += `  // Prepare training data\n`;
  code += `  const xs = tf.tensor2d(xTrain);\n`;
  code += `  // Check if labels need to be one-hot encoded\n`;
  code += `  let ys;\n`;
  code += `  if (Array.isArray(yTrain[0])) {\n`;
  code += `    // Already in suitable format\n`;
  code += `    ys = tf.tensor2d(yTrain);\n`;
  code += `  } else {\n`;
  code += `    // Convert to one-hot encoding for classification tasks\n`;
  code += `    // First, find the number of classes\n`;
  code += `    const numClasses = Math.max(...yTrain) + 1;\n`;
  code += `    if (numClasses > 1) {\n`;
  code += `      // Classification task\n`;
  code += `      ys = tf.oneHot(tf.tensor1d(yTrain, 'int32'), numClasses);\n`;
  code += `    } else {\n`;
  code += `      // Regression task\n`;
  code += `      ys = tf.tensor2d(yTrain, [yTrain.length, 1]);\n`;
  code += `    }\n`;
  code += `  }\n\n`;
  code += `  // Define training configuration\n`;
  code += `  const config = {\n`;
  code += `    batchSize: batchSize,\n`;
  code += `    epochs: epochs,\n`;
  code += `    callbacks: {\n`;
  code += `      onEpochEnd: (epoch, logs) => {\n`;
  code += `        console.log('Epoch ' + (epoch + 1) + ' of ' + epochs);\n`;
  code += `        console.log('  Loss: ' + logs.loss.toFixed(4) + \n`;
  code += `                   (logs.acc ? ', Accuracy: ' + logs.acc.toFixed(4) : ''));\n`;
  code += `      }\n`;
  code += `    }\n`;
  code += `  };\n\n`;
  code += `  // Handle custom models (like linear regression)\n`;
  code += `  if (model.type === 'linear_regression') {\n`;
  code += `    // Basic implementation of linear regression training\n`;
  code += `    const learningRate = model.config.learning_rate;\n`;
  code += `    const inputDim = xTrain[0].length;\n`;
  code += `    \n`;
  code += `    // Initialize weights if not already set\n`;
  code += `    if (!model.weights) {\n`;
  code += `      model.weights = Array(inputDim).fill(0).map(() => Math.random() * 0.1);\n`;
  code += `      model.bias = Math.random() * 0.1;\n`;
  code += `    }\n`;
  code += `    \n`;
  code += `    // Simple gradient descent\n`;
  code += `    const history = { loss: [] };\n`;
  code += `    \n`;
  code += `    for (let epoch = 0; epoch < epochs; epoch++) {\n`;
  code += `      let epochLoss = 0;\n`;
  code += `      \n`;
  code += `      // Process in batches\n`;
  code += `      for (let i = 0; i < xTrain.length; i += batchSize) {\n`;
  code += `        const batchEnd = Math.min(i + batchSize, xTrain.length);\n`;
  code += `        const batchX = xTrain.slice(i, batchEnd);\n`;
  code += `        const batchY = yTrain.slice(i, batchEnd);\n`;
  code += `        \n`;
  code += `        // Forward pass\n`;
  code += `        const predictions = model.forward(batchX);\n`;
  code += `        \n`;
  code += `        // Calculate gradients and update weights\n`;
  code += `        let batchLoss = 0;\n`;
  code += `        const weightGradients = Array(inputDim).fill(0);\n`;
  code += `        let biasGradient = 0;\n`;
  code += `        \n`;
  code += `        for (let j = 0; j < batchX.length; j++) {\n`;
  code += `          const error = predictions[j] - batchY[j];\n`;
  code += `          batchLoss += error * error;\n`;
  code += `          \n`;
  code += `          // Gradients for weights and bias\n`;
  code += `          for (let k = 0; k < inputDim; k++) {\n`;
  code += `            weightGradients[k] += error * batchX[j][k];\n`;
  code += `          }\n`;
  code += `          biasGradient += error;\n`;
  code += `        }\n`;
  code += `        \n`;
  code += `        // Update weights and bias\n`;
  code += `        for (let k = 0; k < inputDim; k++) {\n`;
  code += `          model.weights[k] -= learningRate * weightGradients[k] / batchX.length;\n`;
  code += `        }\n`;
  code += `        model.bias -= learningRate * biasGradient / batchX.length;\n`;
  code += `        \n`;
  code += `        batchLoss /= batchX.length;\n`;
  code += `        epochLoss += batchLoss;\n`;
  code += `      }\n`;
  code += `      \n`;
  code += `      epochLoss /= Math.ceil(xTrain.length / batchSize);\n`;
  code += `      history.loss.push(epochLoss);\n`;
  code += `      \n`;
  code += `      console.log('Epoch ' + (epoch + 1) + ' of ' + epochs + ', Loss: ' + epochLoss.toFixed(4));\n`;
  code += `    }\n`;
  code += `    \n`;
  code += `    return history;\n`;
  code += `  } else {\n`;
  code += `    // TensorFlow.js models\n`;
  code += `    const history = await actualModel.fit(xs, ys, config);\n`;
  code += `    return history;\n`;
  code += `  }\n`;
  code += `}\n\n`;
  code += `let ${historyVar} = await trainModel(${modelVar}, ${xTrainVar}, ${yTrainVar});\n`;
  
  return code;
};

Blockly.JavaScript['evaluate_model'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var xTestVar = Blockly.JavaScript.valueToCode(block, 'X_TEST', Blockly.JavaScript.ORDER_ATOMIC) || 'X_test';
  var yTestVar = Blockly.JavaScript.valueToCode(block, 'Y_TEST', Blockly.JavaScript.ORDER_ATOMIC) || 'y_test';
  var evalVar = block.getFieldValue('EVAL_VAR');
  
  var code = `// Evaluate model\n`;
  code += `async function evaluateModel(model, xTest, yTest) {\n`;
  code += `  // Check model object type\n`;
  code += `  const actualModel = model.model ? model.model : model;\n\n`;
  
  code += `  // Handle custom models (like linear regression)\n`;
  code += `  if (model.type === 'linear_regression') {\n`;
  code += `    // Make predictions\n`;
  code += `    const predictions = model.forward(xTest);\n`;
  code += `    \n`;
  code += `    // Calculate MSE\n`;
  code += `    let mse = 0;\n`;
  code += `    for (let i = 0; i < predictions.length; i++) {\n`;
  code += `      mse += Math.pow(predictions[i] - yTest[i], 2);\n`;
  code += `    }\n`;
  code += `    mse /= predictions.length;\n`;
  code += `    \n`;
  code += `    // Calculate MAE\n`;
  code += `    let mae = 0;\n`;
  code += `    for (let i = 0; i < predictions.length; i++) {\n`;
  code += `      mae += Math.abs(predictions[i] - yTest[i]);\n`;
  code += `    }\n`;
  code += `    mae /= predictions.length;\n`;
  code += `    \n`;
  code += `    // Calculate R²\n`;
  code += `    let meanY = 0;\n`;
  code += `    for (let i = 0; i < yTest.length; i++) {\n`;
  code += `      meanY += yTest[i];\n`;
  code += `    }\n`;
  code += `    meanY /= yTest.length;\n`;
  code += `    \n`;
  code += `    let totalSS = 0;\n`;
  code += `    let residualSS = 0;\n`;
  code += `    for (let i = 0; i < yTest.length; i++) {\n`;
  code += `      totalSS += Math.pow(yTest[i] - meanY, 2);\n`;
  code += `      residualSS += Math.pow(yTest[i] - predictions[i], 2);\n`;
  code += `    }\n`;
  code += `    const rSquared = 1 - (residualSS / totalSS);\n`;
  code += `    \n`;
  code += `    console.log('Model evaluation:');\n`;
  code += `    console.log('  Mean Squared Error (MSE): ' + mse.toFixed(4));\n`;
  code += `    console.log('  Mean Absolute Error (MAE): ' + mae.toFixed(4));\n`;
  code += `    console.log('  R-squared: ' + rSquared.toFixed(4));\n`;
  code += `    \n`;
  code += `    return { mse, mae, rSquared };\n`;
  code += `  } else {\n`;
  code += `    // TensorFlow.js models\n`;
  code += `    // Prepare test data\n`;
  code += `    const xs = tf.tensor2d(xTest);\n`;
  code += `    let ys;\n`;
  code += `    \n`;
  code += `    if (Array.isArray(yTest[0])) {\n`;
  code += `      // Already in suitable format\n`;
  code += `      ys = tf.tensor2d(yTest);\n`;
  code += `    } else {\n`;
  code += `      // Check if it's a classification task\n`;
  code += `      const numClasses = Math.max(...yTest) + 1;\n`;
  code += `      if (numClasses > 1) {\n`;
  code += `        // Classification task\n`;
  code += `        ys = tf.oneHot(tf.tensor1d(yTest, 'int32'), numClasses);\n`;
  code += `      } else {\n`;
  code += `        // Regression task\n`;
  code += `        ys = tf.tensor2d(yTest, [yTest.length, 1]);\n`;
  code += `      }\n`;
  code += `    }\n`;
  code += `    \n`;
  code += `    // Evaluate the model\n`;
  code += `    const result = await actualModel.evaluate(xs, ys);\n`;
  code += `    \n`;
  code += `    // Extract metrics\n`;
  code += `    const metrics = {};\n`;
  code += `    for (let i = 0; i < actualModel.metrics.length; i++) {\n`;
  code += `      metrics[actualModel.metrics[i]] = await result[i].dataSync()[0];\n`;
  code += `    }\n`;
  code += `    \n`;
  code += `    console.log('Model evaluation:');\n`;
  code += `    Object.keys(metrics).forEach(metric => {\n`;
  code += `      console.log('  ' + metric + ': ' + metrics[metric].toFixed(4));\n`;
  code += `    });\n`;
  code += `    \n`;
  code += `    return metrics;\n`;
  code += `  }\n`;
  code += `}\n\n`;
  code += `let ${evalVar} = await evaluateModel(${modelVar}, ${xTestVar}, ${yTestVar});\n`;
  
  return code;
};

Blockly.JavaScript['predict'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var xDataVar = Blockly.JavaScript.valueToCode(block, 'X_DATA', Blockly.JavaScript.ORDER_ATOMIC) || 'X_data';
  var resultVar = block.getFieldValue('RESULT_VAR');
  
  var code = `// Make predictions with the model\n`;
  code += `function predictWithModel(model, xData) {\n`;
  code += `  // Check model object type\n`;
  code += `  const actualModel = model.model ? model.model : model;\n\n`;
  
  code += `  // Handle custom models (like linear regression)\n`;
  code += `  if (model.type === 'linear_regression') {\n`;
  code += `    const predictions = model.forward(xData);\n`;
  code += `    console.log('Predictions made with linear regression model');\n`;
  code += `    return predictions;\n`;
  code += `  } else {\n`;
  code += `    // TensorFlow.js models\n`;
  code += `    const xs = tf.tensor2d(xData);\n`;
  code += `    const predictions = actualModel.predict(xs);\n`;
  code += `    \n`;
  code += `    // Check if it's a classification task\n`;
  code += `    if (predictions.shape[1] > 1) {\n`;
  code += `      // Classification - get class indices\n`;
  code += `      const classIndices = predictions.argMax(1).dataSync();\n`;
  code += `      console.log('Classification predictions made - first few results:');\n`;
  code += `      for (let i = 0; i < Math.min(5, classIndices.length); i++) {\n`;
  code += `        console.log('  Sample ' + i + ': Class ' + classIndices[i]);\n`;
  code += `      }\n`;
  code += `      return Array.from(classIndices);\n`;
  code += `    } else {\n`;
  code += `      // Regression - get values\n`;
  code += `      const values = predictions.dataSync();\n`;
  code += `      console.log('Regression predictions made - first few results:');\n`;
  code += `      for (let i = 0; i < Math.min(5, values.length); i++) {\n`;
  code += `        console.log('  Sample ' + i + ': ' + values[i].toFixed(4));\n`;
  code += `      }\n`;
  code += `      return Array.from(values);\n`;
  code += `    }\n`;
  code += `  }\n`;
  code += `}\n\n`;
  code += `let ${resultVar} = predictWithModel(${modelVar}, ${xDataVar});\n`;
  
  return code;
};

Blockly.JavaScript['plot_data'] = function(block) {
  var xVar = Blockly.JavaScript.valueToCode(block, 'X', Blockly.JavaScript.ORDER_ATOMIC) || '[]';
  var yVar = Blockly.JavaScript.valueToCode(block, 'Y', Blockly.JavaScript.ORDER_ATOMIC) || '[]';
  var plotType = block.getFieldValue('PLOT_TYPE');
  var title = block.getFieldValue('TITLE');
  
  var code = `// Plot data visualization\n`;
  code += `function plotData(x, y, plotType='${plotType}', title='${title}') {\n`;
  code += `  let chartData;\n`;
  code += `  let chartOptions;\n\n`;
  
  code += `  if (plotType === 'scatter') {\n`;
  code += `    // Prepare scatter plot data\n`;
  code += `    const points = [];\n`;
  code += `    for (let i = 0; i < Math.min(x.length, y.length); i++) {\n`;
  code += `      points.push({x: x[i], y: y[i]});\n`;
  code += `    }\n\n`;
  
  code += `    chartData = {\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Data Points',\n`;
  code += `        data: points,\n`;
  code += `        backgroundColor: 'rgba(54, 162, 235, 0.7)',\n`;
  code += `        borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `        pointRadius: 5,\n`;
  code += `        pointHoverRadius: 7\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        xAxes: [{\n`;
  code += `          type: 'linear',\n`;
  code += `          position: 'bottom',\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'X Value'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        yAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Y Value'\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  } else if (plotType === 'line') {\n`;
  code += `    // Prepare line chart data\n`;
  code += `    const labels = Array.from({length: x.length}, (_, i) => i.toString());\n`;
  code += `    \n`;
  code += `    chartData = {\n`;
  code += `      labels: labels,\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Data Series',\n`;
  code += `        data: y,\n`;
  code += `        fill: false,\n`;
  code += `        borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `        tension: 0.1\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        xAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'X Value'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        yAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Y Value'\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  } else if (plotType === 'bar') {\n`;
  code += `    // Prepare bar chart data\n`;
  code += `    const labels = Array.from({length: x.length}, (_, i) => x[i].toString());\n`;
  code += `    \n`;
  code += `    chartData = {\n`;
  code += `      labels: labels,\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Data Values',\n`;
  code += `        data: y,\n`;
  code += `        backgroundColor: 'rgba(54, 162, 235, 0.7)',\n`;
  code += `        borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `        borderWidth: 1\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        yAxes: [{\n`;
  code += `          ticks: {\n`;
  code += `            beginAtZero: true\n`;
  code += `          },\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Value'\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  } else if (plotType === 'histogram') {\n`;
  code += `    // Prepare histogram data\n`;
  code += `    const values = y.length > 0 ? y : x;  // If y is empty, use x for histogram\n`;
  
  code += `    // Calculate bins\n`;
  code += `    const min = Math.min(...values);\n`;
  code += `    const max = Math.max(...values);\n`;
  code += `    const numBins = Math.min(20, Math.ceil(Math.sqrt(values.length)));\n`;
  code += `    const binWidth = (max - min) / numBins;\n`;
  
  code += `    // Initialize bins\n`;
  code += `    const bins = Array(numBins).fill(0);\n`;
  code += `    const binLabels = Array(numBins).fill().map((_, i) => \n`;
  code += `      (min + i * binWidth).toFixed(2) + ' - ' + (min + (i + 1) * binWidth).toFixed(2)\n`;
  code += `    );\n`;
  
  code += `    // Fill bins\n`;
  code += `    values.forEach(value => {\n`;
  code += `      const binIndex = Math.min(numBins - 1, Math.floor((value - min) / binWidth));\n`;
  code += `      bins[binIndex]++;\n`;
  code += `    });\n`;
  
  code += `    chartData = {\n`;
  code += `      labels: binLabels,\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Frequency',\n`;
  code += `        data: bins,\n`;
  code += `        backgroundColor: 'rgba(54, 162, 235, 0.7)',\n`;
  code += `        borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `        borderWidth: 1\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        yAxes: [{\n`;
  code += `          ticks: {\n`;
  code += `            beginAtZero: true\n`;
  code += `          },\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Frequency'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        xAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Value Range'\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  }\n\n`;
  
  code += `  // Create and return the chart\n`;
  code += `  return createChart(plotType === 'scatter' ? 'scatter' : 'bar', chartData, chartOptions);\n`;
  code += `}\n\n`;
  code += `plotData(${xVar}, ${yVar});\n`;
  
  return code;
};

Blockly.JavaScript['plot_history'] = function(block) {
  var historyVar = Blockly.JavaScript.valueToCode(block, 'HISTORY', Blockly.JavaScript.ORDER_ATOMIC) || 'history';
  var metric = block.getFieldValue('METRIC');
  var title = block.getFieldValue('TITLE');
  
  var code = `// Plot training history\n`;
  code += `function plotTrainingHistory(history, metric='${metric}', title='${title}') {\n`;
  code += `  if (!history || (!history.history && !history.loss)) {\n`;
  code += `    console.error('Invalid history object');\n`;
  code += `    return null;\n`;
  code += `  }\n\n`;
  
  code += `  // Extract data from history object\n`;
  code += `  let lossData;\n`;
  code += `  let accuracyData;\n`;
  code += `  \n`;
  code += `  if (history.history) {\n`;
  code += `    // TensorFlow.js history format\n`;
  code += `    lossData = history.history.loss;\n`;
  code += `    accuracyData = history.history.acc || history.history.accuracy;\n`;
  code += `  } else {\n`;
  code += `    // Custom history format\n`;
  code += `    lossData = history.loss;\n`;
  code += `    accuracyData = history.accuracy;\n`;
  code += `  }\n\n`;
  
  code += `  // Create labels for x-axis (epochs)\n`;
  code += `  const epochs = lossData ? lossData.length : 0;\n`;
  code += `  const labels = Array.from({length: epochs}, (_, i) => (i + 1).toString());\n\n`;
  
  code += `  let chartData;\n`;
  code += `  let chartOptions;\n\n`;
  
  code += `  if (metric === 'both' && accuracyData) {\n`;
  code += `    // Plot both loss and accuracy\n`;
  code += `    chartData = {\n`;
  code += `      labels: labels,\n`;
  code += `      datasets: [\n`;
  code += `        {\n`;
  code += `          label: 'Loss',\n`;
  code += `          data: lossData,\n`;
  code += `          fill: false,\n`;
  code += `          borderColor: 'rgba(255, 99, 132, 1)',\n`;
  code += `          tension: 0.1,\n`;
  code += `          yAxisID: 'y-axis-1'\n`;
  code += `        },\n`;
  code += `        {\n`;
  code += `          label: 'Accuracy',\n`;
  code += `          data: accuracyData,\n`;
  code += `          fill: false,\n`;
  code += `          borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `          tension: 0.1,\n`;
  code += `          yAxisID: 'y-axis-2'\n`;
  code += `        }\n`;
  code += `      ]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        xAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Epoch'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        yAxes: [\n`;
  code += `          {\n`;
  code += `            id: 'y-axis-1',\n`;
  code += `            type: 'linear',\n`;
  code += `            position: 'left',\n`;
  code += `            scaleLabel: {\n`;
  code += `              display: true,\n`;
  code += `              labelString: 'Loss'\n`;
  code += `            }\n`;
  code += `          },\n`;
  code += `          {\n`;
  code += `            id: 'y-axis-2',\n`;
  code += `            type: 'linear',\n`;
  code += `            position: 'right',\n`;
  code += `            scaleLabel: {\n`;
  code += `              display: true,\n`;
  code += `              labelString: 'Accuracy'\n`;
  code += `            },\n`;
  code += `            gridLines: {\n`;
  code += `              drawOnChartArea: false\n`;
  code += `            },\n`;
  code += `            ticks: {\n`;
  code += `              min: 0,\n`;
  code += `              max: 1\n`;
  code += `            }\n`;
  code += `          }\n`;
  code += `        ]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  } else if (metric === 'accuracy' && accuracyData) {\n`;
  code += `    // Plot only accuracy\n`;
  code += `    chartData = {\n`;
  code += `      labels: labels,\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Accuracy',\n`;
  code += `        data: accuracyData,\n`;
  code += `        fill: false,\n`;
  code += `        borderColor: 'rgba(54, 162, 235, 1)',\n`;
  code += `        tension: 0.1\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        xAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Epoch'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        yAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Accuracy'\n`;
  code += `          },\n`;
  code += `          ticks: {\n`;
  code += `            min: 0,\n`;
  code += `            max: 1\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  } else {\n`;
  code += `    // Default: plot loss\n`;
  code += `    chartData = {\n`;
  code += `      labels: labels,\n`;
  code += `      datasets: [{\n`;
  code += `        label: 'Loss',\n`;
  code += `        data: lossData,\n`;
  code += `        fill: false,\n`;
  code += `        borderColor: 'rgba(255, 99, 132, 1)',\n`;
  code += `        tension: 0.1\n`;
  code += `      }]\n`;
  code += `    };\n\n`;
  
  code += `    chartOptions = {\n`;
  code += `      responsive: true,\n`;
  code += `      maintainAspectRatio: false,\n`;
  code += `      title: {\n`;
  code += `        display: true,\n`;
  code += `        text: title\n`;
  code += `      },\n`;
  code += `      scales: {\n`;
  code += `        xAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Epoch'\n`;
  code += `          }\n`;
  code += `        }],\n`;
  code += `        yAxes: [{\n`;
  code += `          scaleLabel: {\n`;
  code += `            display: true,\n`;
  code += `            labelString: 'Loss'\n`;
  code += `          }\n`;
  code += `        }]\n`;
  code += `      }\n`;
  code += `    };\n`;
  code += `  }\n\n`;
  
  code += `  // Create and return the chart\n`;
  code += `  return createChart('line', chartData, chartOptions);\n`;
  code += `}\n\n`;
  code += `plotTrainingHistory(${historyVar});\n`;
  
  return code;
};

Blockly.JavaScript['confusion_matrix'] = function(block) {
  var modelVar = Blockly.JavaScript.valueToCode(block, 'MODEL', Blockly.JavaScript.ORDER_ATOMIC) || 'model';
  var xTestVar = Blockly.JavaScript.valueToCode(block, 'X_TEST', Blockly.JavaScript.ORDER_ATOMIC) || 'X_test';
  var yTestVar = Blockly.JavaScript.valueToCode(block, 'Y_TEST', Blockly.JavaScript.ORDER_ATOMIC) || 'y_test';
  var title = block.getFieldValue('TITLE');
  
  var code = `// Plot confusion matrix\n`;
  code += `async function plotConfusionMatrix(model, xTest, yTest, title='${title}') {\n`;
  code += `  // Check if model is valid\n`;
  code += `  if (!model) {\n`;
  code += `    console.error('Invalid model object');\n`;
  code += `    return null;\n`;
  code += `  }\n\n`;
  
  code += `  // Get predictions\n`;
  code += `  let predictions;\n`;
  code += `  if (model.model) {\n`;
  code += `    // Handle neural network container\n`;
  code += `    const xs = tf.tensor2d(xTest);\n`;
  code += `    const preds = model.model.predict(xs);\n`;
  code += `    predictions = Array.from(preds.argMax(1).dataSync());\n`;
  code += `  } else if (model.predict) {\n`;
  code += `    // Standard TensorFlow.js model\n`;
  code += `    const xs = tf.tensor2d(xTest);\n`;
  code += `    const preds = model.predict(xs);\n`;
  code += `    \n`;
  code += `    // Check if it's a classification task\n`;
  code += `    if (preds.shape[1] > 1) {\n`;
  code += `      // Multi-class classification\n`;
  code += `      predictions = Array.from(preds.argMax(1).dataSync());\n`;
  code += `    } else {\n`;
  code += `      // Binary classification\n`;
  code += `      predictions = Array.from(preds.dataSync()).map(p => p > 0.5 ? 1 : 0);\n`;
  code += `    }\n`;
  code += `  } else {\n`;
  code += `    console.error('Model does not support prediction');\n`;
  code += `    return null;\n`;
  code += `  }\n\n`;
  
  code += `  // Extract actual labels\n`;
  code += `  const actual = Array.isArray(yTest[0]) \n`;
  code += `    ? yTest.map(y => y.indexOf(Math.max(...y))) // Convert one-hot to indices\n`;
  code += `    : yTest;\n\n`;
  
  code += `  // Find number of classes\n`;
  code += `  const numClasses = Math.max(Math.max(...predictions), Math.max(...actual)) + 1;\n\n`;
  
  code += `  // Initialize confusion matrix\n`;
  code += `  const matrix = Array(numClasses).fill().map(() => Array(numClasses).fill(0));\n\n`;
  
  code += `  // Fill confusion matrix\n`;
  code += `  for (let i = 0; i < predictions.length; i++) {\n`;
  code += `    matrix[actual[i]][predictions[i]]++;\n`;
  code += `  }\n\n`;
  
  code += `  // Create labels\n`;
  code += `  const labels = Array.from({length: numClasses}, (_, i) => 'Class ' + i);\n\n`;
  
  code += `  // Calculate accuracy\n`;
  code += `  let correct = 0;\n`;
  code += `  for (let i = 0; i < numClasses; i++) {\n`;
  code += `    correct += matrix[i][i];\n`;
  code += `  }\n`;
  code += `  const accuracy = correct / predictions.length;\n\n`;
  
  code += `  // Prepare chart data\n`;
  code += `  const datasets = [];\n`;
  code += `  \n`;
  code += `  for (let i = 0; i < numClasses; i++) {\n`;
  code += `    const rowData = matrix[i].map((value, j) => {\n`;
  code += `      return {\n`;
  code += `        x: j,\n`;
  code += `        y: i,\n`;
  code += `        v: value\n`;
  code += `      };\n`;
  code += `    });\n`;
  code += `    \n`;
  code += `    datasets.push({\n`;
  code += `      label: 'Actual Class ' + i,\n`;
  code += `      data: rowData,\n`;
  code += `      backgroundColor: 'rgba(0, 0, 0, 0)', // Transparent background\n`;
  code += `      borderColor: 'rgba(0, 0, 0, 0)' // Transparent border\n`;
  code += `    });\n`;
  code += `  }\n\n`;
  
  code += `  // Create a canvas for the confusion matrix\n`;
  code += `  const container = document.getElementById('visualizationOutput');\n`;
  code += `  container.innerHTML = '';\n`;
  code += `  \n`;
  code += `  // Create a title element\n`;
  code += `  const titleElement = document.createElement('h4');\n`;
  code += `  titleElement.textContent = title + ' (Accuracy: ' + (accuracy * 100).toFixed(2) + '%)';\n`;
  code += `  titleElement.style.textAlign = 'center';\n`;
  code += `  container.appendChild(titleElement);\n`;
  code += `  \n`;
  code += `  // Create a div for the matrix\n`;
  code += `  const matrixDiv = document.createElement('div');\n`;
  code += `  matrixDiv.style.display = 'flex';\n`;
  code += `  matrixDiv.style.justifyContent = 'center';\n`;
  code += `  matrixDiv.style.marginTop = '20px';\n`;
  code += `  container.appendChild(matrixDiv);\n`;
  code += `  \n`;
  code += `  // Create a table for the confusion matrix\n`;
  code += `  const table = document.createElement('table');\n`;
  code += `  table.style.borderCollapse = 'collapse';\n`;
  code += `  table.style.fontSize = '14px';\n`;
  code += `  matrixDiv.appendChild(table);\n`;
  code += `  \n`;
  code += `  // Create header row\n`;
  code += `  const headerRow = document.createElement('tr');\n`;
  code += `  headerRow.innerHTML = '<th style="padding: 8px; background-color: #f8f9fa;"></th>' +\n`;
  code += `    '<th style="padding: 8px; background-color: #f8f9fa; text-align: center;" colspan="' + numClasses + '">Predicted</th>';\n`;
  code += `  table.appendChild(headerRow);\n`;
  code += `  \n`;
  code += `  // Create subheader row\n`;
  code += `  const subheaderRow = document.createElement('tr');\n`;
  code += `  subheaderRow.innerHTML = '<th style="padding: 8px; background-color: #f8f9fa; text-align: right;">Actual</th>';\n`;
  code += `  \n`;
  code += `  for (let i = 0; i < numClasses; i++) {\n`;
  code += `    subheaderRow.innerHTML += '<th style="padding: 8px; background-color: #f8f9fa; text-align: center;">' + labels[i] + '</th>';\n`;
  code += `  }\n`;
  code += `  table.appendChild(subheaderRow);\n`;
  code += `  \n`;
  code += `  // Create data rows\n`;
  code += `  for (let i = 0; i < numClasses; i++) {\n`;
  code += `    const row = document.createElement('tr');\n`;
  code += `    row.innerHTML = '<th style="padding: 8px; background-color: #f8f9fa; text-align: right;">' + labels[i] + '</th>';\n`;
  code += `    \n`;
  code += `    for (let j = 0; j < numClasses; j++) {\n`;
  code += `      const value = matrix[i][j];\n`;
  code += `      const intensity = value > 0 ? Math.min(1, value / 10) : 0;\n`;
  code += `      \n`;
  code += `      let cellColor;\n`;
  code += `      if (i === j) {\n`;
  code += `        // Correctly classified - use blue\n`;
  code += `        cellColor = 'rgba(54, 162, 235, ' + intensity + ')';\n`;
  code += `      } else {\n`;
  code += `        // Incorrectly classified - use red\n`;
  code += `        cellColor = 'rgba(255, 99, 132, ' + intensity + ')';\n`;
  code += `      }\n`;
  code += `      \n`;
  code += `      row.innerHTML += '<td style="padding: 8px; text-align: center; background-color: ' + cellColor + ';">' + value + '</td>';\n`;
  code += `    }\n`;
  code += `    \n`;
  code += `    table.appendChild(row);\n`;
  code += `  }\n`;
  code += `  \n`;
  code += `  // Add a legend\n`;
  code += `  const legend = document.createElement('div');\n`;
  code += `  legend.style.marginTop = '20px';\n`;
  code += `  legend.style.textAlign = 'center';\n`;
  code += `  legend.innerHTML = \n`;
  code += `    '<div style="display: inline-block; margin-right: 20px;"><span style="display: inline-block; width: 20px; height: 20px; background-color: rgba(54, 162, 235, 0.7); margin-right: 5px;"></span>Correct Predictions</div>' +\n`;
  code += `    '<div style="display: inline-block;"><span style="display: inline-block; width: 20px; height: 20px; background-color: rgba(255, 99, 132, 0.7); margin-right: 5px;"></span>Incorrect Predictions</div>';\n`;
  code += `  container.appendChild(legend);\n`;
  code += `  \n`;
  code += `  console.log('Confusion matrix created with ' + numClasses + ' classes');\n`;
  code += `  console.log('Overall accuracy: ' + (accuracy * 100).toFixed(2) + '%');\n`;
  code += `  \n`;
  code += `  return null; // We're not using Chart.js for this visualization\n`;
  code += `}\n\n`;
  code += `await plotConfusionMatrix(${modelVar}, ${xTestVar}, ${yTestVar});\n`;
  
  return code;
};