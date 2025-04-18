/**
 * 自定义Blockly块定义
 * 用于机器学习操作的块
 */

// 数据处理类块
Blockly.Blocks['load_dataset'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Load Dataset")
        .appendField(new Blockly.FieldDropdown([
          ["Iris", "iris"],
          ["Boston Housing", "boston"],
          ["MNIST", "mnist"],
          ["Fashion MNIST", "fashion_mnist"],
          ["CIFAR10", "cifar10"],
          ["Custom CSV", "custom_csv"]
        ]), "DATASET");
    this.appendDummyInput()
        .appendField("Save as Variable")
        .appendField(new Blockly.FieldTextInput("dataset"), "VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("加载预定义的数据集或自定义CSV文件");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['split_data'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Split Dataset");
    this.appendValueInput("DATASET")
        .setCheck(null)
        .appendField("Dataset");
    this.appendDummyInput()
        .appendField("Test Size")
        .appendField(new Blockly.FieldNumber(0.2, 0, 1, 0.1), "TEST_SIZE");
    this.appendDummyInput()
        .appendField("Random Seed")
        .appendField(new Blockly.FieldNumber(42, 0, 9999, 1), "RANDOM_STATE");
    this.appendDummyInput()
        .appendField("Save as Variable")
        .appendField(new Blockly.FieldTextInput("X_train, X_test, y_train, y_test"), "VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("将数据集分割为训练集和测试集");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['normalize_data'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Normalize Data");
    this.appendValueInput("DATA")
        .setCheck(null)
        .appendField("Data");
    this.appendDummyInput()
        .appendField("Method")
        .appendField(new Blockly.FieldDropdown([
          ["Standardization (StandardScaler)", "standard"],
          ["Min-Max Scaling (MinMaxScaler)", "minmax"],
          ["Robust Scaling (RobustScaler)", "robust"]
        ]), "METHOD");
    this.appendDummyInput()
        .appendField("Save as Variable")
        .appendField(new Blockly.FieldTextInput("X_normalized"), "VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("对数据进行归一化处理");
    this.setHelpUrl("");
  }
};

// 模型构建类块
Blockly.Blocks['create_linear_model'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Create Linear Model")
        .appendField(new Blockly.FieldDropdown([
          ["Linear Regression", "linear_regression"],
          ["Logistic Regression", "logistic_regression"],
          ["Ridge Regression", "ridge"],
          ["Lasso Regression", "lasso"]
        ]), "MODEL_TYPE");
    this.appendDummyInput()
        .appendField("Save as Variable")
        .appendField(new Blockly.FieldTextInput("model"), "VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(290);
    this.setTooltip("创建一个线性模型");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['create_neural_network'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Create Neural Network");
    this.appendDummyInput()
        .appendField("Input Dimension")
        .appendField(new Blockly.FieldNumber(10, 1, 1000, 1), "INPUT_DIM");
    this.appendDummyInput()
        .appendField("Save as Variable")
        .appendField(new Blockly.FieldTextInput("model"), "VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(290);
    this.setTooltip("创建一个神经网络模型");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['add_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Add Neural Network Layer");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("Model");
    this.appendDummyInput()
        .appendField("Layer Type")
        .appendField(new Blockly.FieldDropdown([
          ["Dense (Fully Connected)", "dense"],
          ["Dropout", "dropout"],
          ["Conv2D (2D Convolution)", "conv2d"],
          ["MaxPooling2D (2D Max Pooling)", "maxpool2d"]
        ]), "LAYER_TYPE");
    this.appendDummyInput()
        .appendField("Units/Ratio")
        .appendField(new Blockly.FieldNumber(32, 1, 1000, 1), "UNITS");
    this.appendDummyInput()
        .appendField("Activation Function")
        .appendField(new Blockly.FieldDropdown([
          ["ReLU", "relu"],
          ["Sigmoid", "sigmoid"],
          ["Tanh", "tanh"],
          ["Softmax", "softmax"],
          ["None", "none"]
        ]), "ACTIVATION");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(290);
    this.setTooltip("向神经网络模型添加一个层");
    this.setHelpUrl("");
  }
};

// 训练与评估类块
Blockly.Blocks['train_model'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Train Model");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("Model");
    this.appendValueInput("X_TRAIN")
        .setCheck(null)
        .appendField("Training Data X");
    this.appendValueInput("Y_TRAIN")
        .setCheck(null)
        .appendField("Training Labels y");
    this.appendDummyInput()
        .appendField("Batch Size")
        .appendField(new Blockly.FieldNumber(32, 1, 1000, 1), "BATCH_SIZE");
    this.appendDummyInput()
        .appendField("Epochs")
        .appendField(new Blockly.FieldNumber(10, 1, 1000, 1), "EPOCHS");
    this.appendDummyInput()
        .appendField("Save Training History As")
        .appendField(new Blockly.FieldTextInput("history"), "HISTORY_VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(160);
    this.setTooltip("使用训练数据训练模型");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['evaluate_model'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Evaluate Model");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("Model");
    this.appendValueInput("X_TEST")
        .setCheck(null)
        .appendField("Test Data X");
    this.appendValueInput("Y_TEST")
        .setCheck(null)
        .appendField("Test Labels y");
    this.appendDummyInput()
        .appendField("Save Evaluation Results As")
        .appendField(new Blockly.FieldTextInput("evaluation"), "EVAL_VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(160);
    this.setTooltip("使用测试数据评估模型性能");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['predict'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Predict");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("Model");
    this.appendValueInput("X_DATA")
        .setCheck(null)
        .appendField("Prediction Data X");
    this.appendDummyInput()
        .appendField("Save Predictions As")
        .appendField(new Blockly.FieldTextInput("predictions"), "RESULT_VAR");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(160);
    this.setTooltip("使用模型进行预测");
    this.setHelpUrl("");
  }
};

// 可视化类块
Blockly.Blocks['plot_data'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Plot Data");
    this.appendValueInput("X")
        .setCheck(null)
        .appendField("X Data");
    this.appendValueInput("Y")
        .setCheck(null)
        .appendField("Y Data");
    this.appendDummyInput()
        .appendField("Chart Type")
        .appendField(new Blockly.FieldDropdown([
          ["Scatter Plot", "scatter"],
          ["Line Chart", "line"],
          ["Bar Chart", "bar"],
          ["Histogram", "histogram"]
        ]), "PLOT_TYPE");
    this.appendDummyInput()
        .appendField("Title")
        .appendField(new Blockly.FieldTextInput("Data Visualization"), "TITLE");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(65);
    this.setTooltip("绘制数据可视化图表");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['plot_history'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Plot Training History");
    this.appendValueInput("HISTORY")
        .setCheck(null)
        .appendField("Training History");
    this.appendDummyInput()
        .appendField("Metric")
        .appendField(new Blockly.FieldDropdown([
          ["Loss", "loss"],
          ["Accuracy", "accuracy"],
          ["Loss and Accuracy", "both"]
        ]), "METRIC");
    this.appendDummyInput()
        .appendField("Title")
        .appendField(new Blockly.FieldTextInput("Training History"), "TITLE");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(65);
    this.setTooltip("Plot model training history charts");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['confusion_matrix'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Plot Confusion Matrix");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("Model");
    this.appendValueInput("X_TEST")
        .setCheck(null)
        .appendField("Test Data X");
    this.appendValueInput("Y_TEST")
        .setCheck(null)
        .appendField("Test Labels y");
    this.appendDummyInput()
        .appendField("Title")
        .appendField(new Blockly.FieldTextInput("Confusion Matrix"), "TITLE");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(65);
    this.setTooltip("Plot confusion matrix for classification models");
    this.setHelpUrl("");
  }
};