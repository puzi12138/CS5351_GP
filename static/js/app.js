/**
 * 应用程序主逻辑
 * 负责初始化Blockly工作区、处理用户交互和执行生成的代码
 */

// 全局变量
let workspace = null;
let runningCode = false;
let currentChart = null;

// 初始化函数
document.addEventListener('DOMContentLoaded', function() {
  initBlockly();
  initUI();
});

/**
 * 初始化Blockly工作区
 */
function initBlockly() {
  // 配置工作区
  const blocklyArea = document.getElementById('blocklyDiv');
  
  // 创建工作区
  workspace = Blockly.inject(blocklyArea, {
    toolbox: document.getElementById('toolbox'),
    grid: {
      spacing: 20,
      length: 3,
      colour: '#ccc',
      snap: true
    },
    zoom: {
      controls: true,
      wheel: true,
      startScale: 1.0,
      maxScale: 3,
      minScale: 0.3,
      scaleSpeed: 1.2
    },
    trashcan: true
  });

  // 添加事件监听器
  workspace.addChangeListener(onWorkspaceChange);
}

/**
 * 初始化UI元素和事件处理
 */
function initUI() {
  // 运行按钮
  document.getElementById('runModel').addEventListener('click', runCode);
  
  // 停止按钮
  document.getElementById('stopModel').addEventListener('click', stopCode);
  
  // 查看代码按钮
  document.getElementById('viewCode').addEventListener('click', showGeneratedCode);
  
  // 查看数据按钮
  document.getElementById('viewData').addEventListener('click', showDataViewer);
  
  // 新建项目按钮
  document.getElementById('newProject').addEventListener('click', newProject);
  
  // 保存项目按钮
  document.getElementById('saveProject').addEventListener('click', saveProject);
  
  // 加载项目按钮
  document.getElementById('loadProject').addEventListener('click', loadProject);
  
  // 帮助按钮
  document.getElementById('helpButton').addEventListener('click', showHelp);
  
  // 复制代码按钮
  document.getElementById('copyCode').addEventListener('click', copyGeneratedCode);
}

/**
 * 工作区变化事件处理
 */
function onWorkspaceChange(event) {
  // 当工作区发生变化时更新代码预览
  if (event.type === Blockly.Events.BLOCK_MOVE ||
      event.type === Blockly.Events.BLOCK_CREATE ||
      event.type === Blockly.Events.BLOCK_DELETE ||
      event.type === Blockly.Events.BLOCK_CHANGE) {
    updateCodePreview();
  }
}

/**
 * 更新代码预览
 */
function updateCodePreview() {
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  document.getElementById('codeOutput').textContent = code;
}

/**
 * 运行生成的代码
 */
async function runCode() {
  if (runningCode) {
    console.log('代码已在运行中...');
    return;
  }
  
  // 清空控制台和可视化输出
  clearOutputs();
  
  // 获取生成的代码
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  if (!code.trim()) {
    logToConsole('Error: No executable code. Please drag blocks to the workspace.');
    return;
  }
  
  // 设置运行状态
  runningCode = true;
  logToConsole('Starting code execution...');
  
  try {
    // 包装代码为异步函数并执行
    const asyncFunction = new Function(`
      return (async function() {
        try {
          ${code}
          return "Execution completed";
        } catch (error) {
          return "Execution error: " + error.message;
        }
      })();
    `);
    
    const result = await asyncFunction();
    logToConsole(result);
  } catch (error) {
    logToConsole(`Execution error: ${error.message}`);
    console.error(error);
  } finally {
    runningCode = false;
  }
}

/**
 * 停止代码执行
 */
function stopCode() {
  if (!runningCode) {
    logToConsole('No code is currently running.');
    return;
  }
  
  // 在实际应用中，这里应该有更复杂的逻辑来停止异步操作
  // 但在简单实现中，我们只设置标志位
  runningCode = false;
  logToConsole('Code execution has been stopped.');
  
  // 释放PyTorch资源
  // 注意：在使用ONNX.js时，需要不同的资源释放方式
  // 这里需要根据实际使用的PyTorch.js或ONNX.js实现来调整
  console.log('Model execution stopped and resources released');
}

/**
 * 显示生成的代码
 */
function showGeneratedCode() {
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  document.getElementById('modalCodeOutput').textContent = code;
  $('#codeModal').modal('show');
}

/**
 * 复制生成的代码到剪贴板
 */
function copyGeneratedCode() {
  const codeText = document.getElementById('modalCodeOutput').textContent;
  navigator.clipboard.writeText(codeText).then(() => {
    alert('Code copied to clipboard!');
  }).catch(err => {
    console.error('Copy failed:', err);
    alert('Copy failed, please manually select and copy the code.');
  });
}

/**
 * 显示数据查看器
 */
function showDataViewer() {
  // 这里应该实现数据可视化功能
  // 在简单实现中，我们只显示一个提示
  alert('Data viewer functionality is under development...');
}

/**
 * 新建项目
 */
function newProject() {
  if (confirm('确定要创建新项目吗？当前未保存的内容将丢失。')) {
    workspace.clear();
    clearOutputs();
  }
}

/**
 * 保存项目
 */
function saveProject() {
  try {
    // 获取工作区XML
    const xml = Blockly.Xml.workspaceToDom(workspace);
    const xmlText = Blockly.Xml.domToText(xml);
    
    // 创建Blob对象
    const blob = new Blob([xmlText], {type: 'application/xml'});
    const url = URL.createObjectURL(blob);
    
    // 创建下载链接
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ml_project.xml';
    a.click();
    
    // 释放URL对象
    URL.revokeObjectURL(url);
    
    logToConsole('项目已保存。');
  } catch (error) {
    console.error('保存项目失败:', error);
    logToConsole(`保存项目失败: ${error.message}`);
  }
}

/**
 * 加载项目
 */
function loadProject() {
  // 创建文件输入元素
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.xml';
  
  fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      try {
        const xmlText = e.target.result;
        const xml = Blockly.Xml.textToDom(xmlText);
        
        // 清空当前工作区并加载新项目
        workspace.clear();
        Blockly.Xml.domToWorkspace(xml, workspace);
        
        logToConsole('项目已加载。');
      } catch (error) {
        console.error('加载项目失败:', error);
        logToConsole(`加载项目失败: ${error.message}`);
      }
    };
    
    reader.readAsText(file);
  });
  
  fileInput.click();
}

/**
 * 显示帮助信息
 */
function showHelp() {
  $('#helpModal').modal('show');
}

/**
 * 清空所有输出区域
 */
function clearOutputs() {
  // 清空控制台输出
  document.getElementById('consoleOutput').textContent = '';
  
  // 清空可视化区域
  const visualizationOutput = document.getElementById('visualizationOutput');
  visualizationOutput.innerHTML = '';
  
  // 销毁当前图表
  if (currentChart) {
    currentChart.destroy();
    currentChart = null;
  }
}

/**
 * 记录消息到控制台输出
 */
function logToConsole(message) {
  const consoleOutput = document.getElementById('consoleOutput');
  const timestamp = new Date().toLocaleTimeString();
  consoleOutput.textContent += `[${timestamp}] ${message}\n`;
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

/**
 * 创建图表
 */
function createChart(type, data, options, container = 'visualizationOutput') {
  // 销毁现有图表
  if (currentChart) {
    currentChart.destroy();
  }
  
  // 创建画布
  const canvas = document.createElement('canvas');
  document.getElementById(container).innerHTML = '';
  document.getElementById(container).appendChild(canvas);
  
  // 创建图表
  const ctx = canvas.getContext('2d');
  currentChart = new Chart(ctx, {
    type: type,
    data: data,
    options: options
  });
  
  return currentChart;
}

// 将常用函数暴露给全局作用域，以便生成的代码可以使用
window.logToConsole = logToConsole;
window.createChart = createChart;

// 重写console.log以将输出重定向到UI
const originalConsoleLog = console.log;
console.log = function() {
  // 调用原始console.log
  originalConsoleLog.apply(console, arguments);
  
  // 将输出重定向到UI
  const message = Array.from(arguments).map(arg => {
    if (typeof arg === 'object') {
      return JSON.stringify(arg, null, 2);
    } else {
      return String(arg);
    }
  }).join(' ');
  
  logToConsole(message);
};