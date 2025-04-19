<p align="center">
  <a href="https://blocklyml.onrender.com/">
<img src="https://raw.githubusercontent.com/chekoduadarsh/BlocklyML/main/media/blocklyML_Banner.png" height="80" />
    </a>
    </p>
<p align="center">
<a href="https://blocklyml.onrender.com/">SnapML</a>
</p>
<p align="center">
<img src="https://img.shields.io/github/license/chekoduadarsh/BlocklyML">
<img src="https://img.shields.io/github/issues/chekoduadarsh/BlocklyML">
<img src="https://img.shields.io/github/last-commit/chekoduadarsh/BlocklyML">
 <img src="https://github.com/chekoduadarsh/BlocklyML/actions/workflows/codeql.yml/badge.svg">
   </p>

# SnapML

SnapML is a visual programming tool designed for software developers to streamline machine learning code development. It transforms the traditional coding process into an intuitive drag-and-drop experience, similar to building with Lego blocks.

## Features

### 🔧 Developer-Centric Design
- Visual programming interface for rapid development
- Real-time Python code generation
- Integrated development tools (version control, debugging)
- Support for team collaboration

### 📦 Pre-built Module Library
- Data processing modules
- Model training modules
- Evaluation modules
- Custom module creation capability

### 🛠 Framework Support
- scikit-learn integration
- PyCaret integration
- Easy extension for other ML frameworks

### 💻 Code Generation
- Clean, optimized Python code
- Export to Python scripts
- Export to Jupyter notebooks
- Code formatting and optimization

## Getting Started

### Prerequisites
- Python 3.11+ (推荐使用 Python 3.11.11)
- conda environment manager
- pycaret 3.3.2 及其依赖
- 其他依赖请参考 requirements.txt

### 环境配置
推荐使用以下已配置好的环境：
```bash
# 已配置好的环境路径
环境名称：py311
路径：/home/chengwei/anaconda3/envs/py311

# 激活环境
conda activate py311
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SnapML.git

# 方法1：使用现有的 py311 环境（推荐）
conda activate py311

# 方法2：创建新的 conda 环境（如果需要）
conda create -n py311 python=3.11
conda activate py311

# 安装 pycaret（如果尚未安装）
pip install pycaret==3.3.2

# 安装其他依赖
pip install -r requirements.txt

# 启动应用
cd SnapML
python app.py
```

应用将在 http://localhost:5005 启动。如果端口 5005 被占用，可以使用以下命令清理：
```bash
# 查找占用端口的进程
lsof -i :5005

# 终止占用端口的进程（将 PID 替换为实际进程ID）
kill -9 <PID>
```

### Quick Start
1. Access the web interface at `http://localhost:5005`
2. Use the left toolbox to select ML components
3. Drag and drop components to build your workflow
4. Connect components to create logical flow
5. View generated code in real-time
6. Export your code when ready

## Development

### Project Structure
```
SnapML/
├── app.py              # Main application entry
├── static/            # Static assets
│   ├── css/          # Stylesheets
│   ├── js/           # JavaScript files
│   └── media/        # Media files
├── templates/         # HTML templates
├── libs/             # Core libraries
└── tests/            # Test files
```

