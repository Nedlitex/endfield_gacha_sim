@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   终末地抽卡策略模拟器 - 一键启动
echo ========================================
echo.

:: Check if Python is installed
echo [1/4] 检查 Python 环境...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set pyver=%%v
echo 检测到 Python %pyver%

:: Check if venv exists
echo.
echo [2/4] 检查虚拟环境...
if not exist "venv\Scripts\activate.bat" (
    echo 创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在
)

:: Activate venv
echo.
echo [3/4] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat

:: Install requirements
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [错误] 安装依赖失败
    pause
    exit /b 1
)
echo 依赖安装完成

:: Run streamlit
echo.
echo [4/4] 启动应用...
echo ========================================
echo 应用将在浏览器中打开
echo 如未自动打开，请访问 http://localhost:8501
echo 按 Ctrl+C 停止应用
echo ========================================
echo.

streamlit run app.py

pause
