/**
 * 自定义Blockly块定义
 * 用于机器学习操作的块
 */

// 数据处理类块
Blockly.Blocks['load_dataset'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("加载数据集")
        .appendField(new Blockly.FieldDropdown([
          ["Iris", "iris"],
          ["Boston Housing", "boston"],
          ["MNIST", "mnist"],
          ["Fashion MNIST", "fashion_mnist"],
          ["CIFAR10", "cifar10"],
          ["自定义CSV", "custom_csv"]
        ]), "DATASET");
    this.appendDummyInput()
        .appendField("保存为变量")
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
        .appendField("分割数据集");
    this.appendValueInput("DATASET")
        .setCheck(null)
        .appendField("数据集");
    this.appendDummyInput()
        .appendField("测试集比例")
        .appendField(new Blockly.FieldNumber(0.2, 0, 1, 0.1), "TEST_SIZE");
    this.appendDummyInput()
        .appendField("随机种子")
        .appendField(new Blockly.FieldNumber(42, 0, 9999, 1), "RANDOM_STATE");
    this.appendDummyInput()
        .appendField("保存为变量")
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
        .appendField("归一化数据");
    this.appendValueInput("DATA")
        .setCheck(null)
        .appendField("数据");
    this.appendDummyInput()
        .appendField("方法")
        .appendField(new Blockly.FieldDropdown([
          ["标准化 (StandardScaler)", "standard"],
          ["最小最大缩放 (MinMaxScaler)", "minmax"],
          ["鲁棒缩放 (RobustScaler)", "robust"]
        ]), "METHOD");
    this.appendDummyInput()
        .appendField("保存为变量")
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
        .appendField("创建线性模型")
        .appendField(new Blockly.FieldDropdown([
          ["线性回归", "linear_regression"],
          ["逻辑回归", "logistic_regression"],
          ["岭回归", "ridge"],
          ["Lasso回归", "lasso"]
        ]), "MODEL_TYPE");
    this.appendDummyInput()
        .appendField("保存为变量")
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
        .appendField("创建神经网络模型");
    this.appendDummyInput()
        .appendField("输入维度")
        .appendField(new Blockly.FieldNumber(10, 1, 1000, 1), "INPUT_DIM");
    this.appendDummyInput()
        .appendField("保存为变量")
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
        .appendField("添加神经网络层");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("模型");
    this.appendDummyInput()
        .appendField("层类型")
        .appendField(new Blockly.FieldDropdown([
          ["Dense (全连接)", "dense"],
          ["Dropout", "dropout"],
          ["Conv2D (二维卷积)", "conv2d"],
          ["MaxPooling2D (二维最大池化)", "maxpool2d"]
        ]), "LAYER_TYPE");
    this.appendDummyInput()
        .appendField("神经元数量/比例")
        .appendField(new Blockly.FieldNumber(32, 1, 1000, 1), "UNITS");
    this.appendDummyInput()
        .appendField("激活函数")
        .appendField(new Blockly.FieldDropdown([
          ["ReLU", "relu"],
          ["Sigmoid", "sigmoid"],
          ["Tanh", "tanh"],
          ["Softmax", "softmax"],
          ["无", "none"]
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
        .appendField("训练模型");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("模型");
    this.appendValueInput("X_TRAIN")
        .setCheck(null)
        .appendField("训练数据 X");
    this.appendValueInput("Y_TRAIN")
        .setCheck(null)
        .appendField("训练标签 y");
    this.appendDummyInput()
        .appendField("批次大小")
        .appendField(new Blockly.FieldNumber(32, 1, 1000, 1), "BATCH_SIZE");
    this.appendDummyInput()
        .appendField("训练轮数")
        .appendField(new Blockly.FieldNumber(10, 1, 1000, 1), "EPOCHS");
    this.appendDummyInput()
        .appendField("保存训练历史为")
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
        .appendField("评估模型");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("模型");
    this.appendValueInput("X_TEST")
        .setCheck(null)
        .appendField("测试数据 X");
    this.appendValueInput("Y_TEST")
        .setCheck(null)
        .appendField("测试标签 y");
    this.appendDummyInput()
        .appendField("保存评估结果为")
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
        .appendField("预测");
    this.appendValueInput("MODEL")
        .setCheck(null)
        .appendField("模型");
    this.appendValueInput("X")
        .setCheck(null)
        .appendField("数据 X");
    this.appendDummyInput()
        .appendField("保存预测结果为")
        .appendField(new Blockly.FieldTextInput("predictions"), "PRED_VAR");
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
        .appendField("绘制数据");
    this.appendValueInput("X")
        .setCheck(null)
        .appendField("X数据");
    this.appendValueInput("Y")
        .setCheck(null)
        .appendField("Y数据");
    this.appendDummyInput()
        .appendField("图表类型")
        .appendField(new Blockly.FieldDropdown([
          ["散点图", "scatter"],
          ["线图", "line"],
          ["柱状图", "bar"],
          ["直方图", "histogram"]
        ]), "PLOT_TYPE");
    this.appendDummyInput()
        .appendField("标题")
        .appendField(new Blockly.FieldTextInput("数据可视化"), "TITLE");
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
        .appendField("绘制训练历史");
    this.appendValueInput("HISTORY")
        .setCheck(null)
        .appendField("训练历史");
    this.appendDummyInput()
        .appendField("指标")
        .appendField(new Blockly.FieldDropdown([
          ["损失 (loss)", "loss"],
          ["准确率 (accuracy)", "accuracy"],
          ["全部", "all"]
        ]), "METRIC");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(65);
    this.setTooltip("绘制模型训练历史图表");
    this.setHelpUrl("");
  }
};

Blockly.Blocks['confusion_matrix'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("绘制混淆矩阵");
    this.appendValueInput("Y_TRUE")
        .setCheck(null)
        .appendField("真实标签");
    this.appendValueInput("Y_PRED")
        .setCheck(null)
        .appendField("预测标签");
    this.setOutput(false);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(65);
    this.setTooltip("绘制分类模型的混淆矩阵");
    this.setHelpUrl("");
  }
};