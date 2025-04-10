/**
 * Application main logic
 * Responsible for initializing Blockly workspace, handling user interactions, and executing generated code
 */

// Global variables
let workspace = null;
let runningCode = false;
let currentChart = null;
let currentDataset = null;

// Initialization function
document.addEventListener('DOMContentLoaded', function() {
  initBlockly();
  initUI();
});

/**
 * Initialize Blockly workspace
 */
function initBlockly() {
  // Configure workspace
  const blocklyArea = document.getElementById('blocklyDiv');
  
  // Create workspace
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

  // Add event listeners
  workspace.addChangeListener(onWorkspaceChange);
}

/**
 * Initialize UI elements and event handlers
 */
function initUI() {
  // Run button
  document.getElementById('runModel').addEventListener('click', runCode);
  
  // Stop button
  document.getElementById('stopModel').addEventListener('click', stopCode);
  
  // View code button
  document.getElementById('viewCode').addEventListener('click', showGeneratedCode);
  
  // View data button
  document.getElementById('viewData').addEventListener('click', showDataViewer);
  
  // New project button
  document.getElementById('newProject').addEventListener('click', newProject);
  
  // Save project button
  document.getElementById('saveProject').addEventListener('click', saveProject);
  
  // Load project button
  document.getElementById('loadProject').addEventListener('click', loadProject);
  
  // Help button
  document.getElementById('helpButton').addEventListener('click', showHelp);
  
  // Copy code button
  document.getElementById('copyCode').addEventListener('click', copyGeneratedCode);
}

/**
 * Workspace change event handler
 */
function onWorkspaceChange(event) {
  // Update code preview when workspace changes
  if (event.type === Blockly.Events.BLOCK_MOVE ||
      event.type === Blockly.Events.BLOCK_CREATE ||
      event.type === Blockly.Events.BLOCK_DELETE ||
      event.type === Blockly.Events.BLOCK_CHANGE) {
    updateCodePreview();
  }
}

/**
 * Update code preview
 */
function updateCodePreview() {
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  document.getElementById('codeOutput').textContent = code;
}

/**
 * Run generated code
 */
async function runCode() {
  if (runningCode) {
    console.log('Code is already running...');
    return;
  }
  
  // Clear console and visualization output
  clearOutputs();
  
  // Get generated code
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  if (!code.trim()) {
    logToConsole('Error: No executable code. Please drag blocks to the workspace.');
    return;
  }
  
  // Set running state
  runningCode = true;
  logToConsole('Starting code execution...');
  
  try {
    // Wrap code as async function and execute
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
 * Stop code execution
 */
function stopCode() {
  if (!runningCode) {
    logToConsole('No code is currently running.');
    return;
  }
  
  // In a real application, there should be more complex logic to stop async operations
  // But in a simple implementation, we just set the flag
  runningCode = false;
  logToConsole('Code execution has been stopped.');
  
  // Release PyTorch resources
  // Note: When using ONNX.js, a different resource release method is needed
  // This needs to be adjusted according to the actual PyTorch.js or ONNX.js implementation used
  console.log('Model execution stopped and resources released');
}

/**
 * Show generated code
 */
function showGeneratedCode() {
  const code = Blockly.JavaScript.workspaceToCode(workspace);
  document.getElementById('modalCodeOutput').textContent = code;
  $('#codeModal').modal('show');
}

/**
 * Copy generated code to clipboard
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
 * Show data viewer
 */
function showDataViewer() {
  // Data visualization implementation
  if (!currentDataset) {
    logToConsole('No dataset loaded. Please load a dataset first.');
    // Switch to the console tab to show the message
    $('#outputTabs a[href="#console"]').tab('show');
    return;
  }
  
  // Switch to visualization tab
  $('#outputTabs a[href="#visualization"]').tab('show');
  
  // Clear previous visualizations
  const visualizationOutput = document.getElementById('visualizationOutput');
  visualizationOutput.innerHTML = '';
  
  // Create a container for the visualization
  const container = document.createElement('div');
  container.style.width = '100%';
  container.style.height = '100%';
  visualizationOutput.appendChild(container);
  
  try {
    // Check what kind of data we have
    if (currentDataset.features && currentDataset.features.length > 0) {
      // Create data summary
      const summaryDiv = document.createElement('div');
      summaryDiv.classList.add('data-summary');
      summaryDiv.innerHTML = `
        <h4>Dataset Summary</h4>
        <p>Number of samples: ${currentDataset.features.length}</p>
        <p>Number of features: ${Array.isArray(currentDataset.features[0]) ? currentDataset.features[0].length : 'N/A'}</p>
      `;
      container.appendChild(summaryDiv);
      
      // Create canvas for visualization
      const canvas = document.createElement('canvas');
      canvas.id = 'dataChart';
      canvas.style.marginTop = '20px';
      container.appendChild(canvas);
      
      // If the data is appropriate for scatter plot (2D)
      if (Array.isArray(currentDataset.features[0]) && currentDataset.features[0].length >= 2) {
        // Prepare data for scatter plot (first two dimensions)
        const scatterData = {
          datasets: [{
            label: 'Dataset Points',
            data: currentDataset.features.map((feature, i) => ({
              x: feature[0],
              y: feature[1],
              class: currentDataset.labels ? currentDataset.labels[i] : null
            })),
            backgroundColor: currentDataset.labels 
              ? currentDataset.features.map((_, i) => {
                  // Generate colors based on label
                  const label = currentDataset.labels[i];
                  const colors = ['rgba(255, 99, 132, 0.7)', 'rgba(54, 162, 235, 0.7)', 
                                 'rgba(255, 206, 86, 0.7)', 'rgba(75, 192, 192, 0.7)', 
                                 'rgba(153, 102, 255, 0.7)', 'rgba(255, 159, 64, 0.7)'];
                  return colors[label % colors.length];
                })
              : 'rgba(54, 162, 235, 0.7)',
            pointRadius: 5,
            pointHoverRadius: 7
          }]
        };
        
        // Create scatter plot
        new Chart(canvas.getContext('2d'), {
          type: 'scatter',
          data: scatterData,
          options: {
            responsive: true,
            maintainAspectRatio: false,
            title: {
              display: true,
              text: 'Dataset Visualization (First Two Dimensions)'
            },
            scales: {
              xAxes: [{
                scaleLabel: {
                  display: true,
                  labelString: 'Feature 1'
                }
              }],
              yAxes: [{
                scaleLabel: {
                  display: true,
                  labelString: 'Feature 2'
                }
              }]
            },
            tooltips: {
              callbacks: {
                label: function(tooltipItem, data) {
                  const dataPoint = data.datasets[0].data[tooltipItem.index];
                  let label = `(${dataPoint.x.toFixed(2)}, ${dataPoint.y.toFixed(2)})`;
                  if (dataPoint.class !== null) {
                    label += `, Class: ${dataPoint.class}`;
                  }
                  return label;
                }
              }
            }
          }
        });
      } else {
        // For non-2D data, create a bar chart of the first few samples
        const numSamples = Math.min(10, currentDataset.features.length);
        const barData = {
          labels: Array.from({length: numSamples}, (_, i) => `Sample ${i+1}`),
          datasets: [{
            label: 'Feature Values',
            data: currentDataset.features.slice(0, numSamples).map(feature => {
              // If feature is an array, return its first value
              return Array.isArray(feature) ? feature[0] : feature;
            }),
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }]
        };
        
        // Create bar chart
        new Chart(canvas.getContext('2d'), {
          type: 'bar',
          data: barData,
          options: {
            responsive: true,
            maintainAspectRatio: false,
            title: {
              display: true,
              text: 'Dataset Preview (First 10 Samples)'
            },
            scales: {
              yAxes: [{
                scaleLabel: {
                  display: true,
                  labelString: 'Value'
                }
              }]
            }
          }
        });
      }
    } else {
      container.innerHTML = '<div class="alert alert-warning">Dataset format not supported for visualization</div>';
    }
  } catch (error) {
    console.error('Visualization error:', error);
    container.innerHTML = `<div class="alert alert-danger">Error creating visualization: ${error.message}</div>`;
  }
}

/**
 * Create new project
 */
function newProject() {
  if (confirm('Are you sure you want to create a new project? Any unsaved changes will be lost.')) {
    workspace.clear();
    clearOutputs();
    currentDataset = null;
  }
}

/**
 * Save project
 */
function saveProject() {
  try {
    // Get workspace XML
    const xml = Blockly.Xml.workspaceToDom(workspace);
    const xmlText = Blockly.Xml.domToText(xml);
    
    // Create Blob object
    const blob = new Blob([xmlText], {type: 'application/xml'});
    const url = URL.createObjectURL(blob);
    
    // Create download link
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ml_project.xml';
    a.click();
    
    // Release URL object
    URL.revokeObjectURL(url);
    
    logToConsole('Project saved successfully.');
  } catch (error) {
    console.error('Failed to save project:', error);
    logToConsole(`Failed to save project: ${error.message}`);
  }
}

/**
 * Load project
 */
function loadProject() {
  // Create file input element
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
        
        // Clear current workspace and load new project
        workspace.clear();
        Blockly.Xml.domToWorkspace(xml, workspace);
        
        logToConsole('Project loaded successfully.');
      } catch (error) {
        console.error('Failed to load project:', error);
        logToConsole(`Failed to load project: ${error.message}`);
      }
    };
    
    reader.readAsText(file);
  });
  
  fileInput.click();
}

/**
 * Show help information
 */
function showHelp() {
  $('#helpModal').modal('show');
}

/**
 * Clear all output areas
 */
function clearOutputs() {
  // Clear console output
  document.getElementById('consoleOutput').textContent = '';
  
  // Clear visualization area
  const visualizationOutput = document.getElementById('visualizationOutput');
  visualizationOutput.innerHTML = '';
  
  // Destroy current chart
  if (currentChart) {
    currentChart.destroy();
    currentChart = null;
  }
}

/**
 * Log message to console output
 */
function logToConsole(message) {
  const consoleOutput = document.getElementById('consoleOutput');
  const timestamp = new Date().toLocaleTimeString();
  consoleOutput.textContent += `[${timestamp}] ${message}\n`;
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

/**
 * Create chart
 */
function createChart(type, data, options, container = 'visualizationOutput') {
  // Destroy existing chart
  if (currentChart) {
    currentChart.destroy();
  }
  
  // Create canvas
  const canvas = document.createElement('canvas');
  document.getElementById(container).innerHTML = '';
  document.getElementById(container).appendChild(canvas);
  
  // Create chart
  const ctx = canvas.getContext('2d');
  currentChart = new Chart(ctx, {
    type: type,
    data: data,
    options: options
  });
  
  return currentChart;
}

// Set dataset globally so it can be used by visualization
function setCurrentDataset(dataset) {
  currentDataset = dataset;
  return dataset;
}

// Expose commonly used functions to global scope so generated code can use them
window.logToConsole = logToConsole;
window.createChart = createChart;
window.setCurrentDataset = setCurrentDataset;

// Override console.log to redirect output to UI
const originalConsoleLog = console.log;
console.log = function() {
  // Call original console.log
  originalConsoleLog.apply(console, arguments);
  
  // Redirect output to UI
  const message = Array.from(arguments).map(arg => {
    if (typeof arg === 'object') {
      return JSON.stringify(arg, null, 2);
    } else {
      return String(arg);
    }
  }).join(' ');
  
  logToConsole(message);
};