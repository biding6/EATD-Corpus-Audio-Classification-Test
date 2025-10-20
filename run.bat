@echo off

REM =================================================
REM ==  EATD-Corpus 抑郁症检测实验 - 一键启动脚本  ==
REM =================================================

REM 提示用户进行选择
ECHO.
ECHO Please select an experiment to run:
ECHO.
ECHO   1. CLAP
ECHO   2. Qwen2-Audio
ECHO   3. Audio_Flamingo_2
ECHO.

SET /p CHOICE="Enter your choice (1, 2, or 3) and press Enter: "

REM --- 核心修正：使用正确的 Conda 环境名称 'CLAP01' ---
ECHO.
ECHO Activating Conda environment 'CLAP01'...
CALL conda activate CLAP01

REM 检查 Conda 环境是否激活成功
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to activate Conda environment. Please check if Conda is installed and the 'CLAP01' environment exists.
    GOTO :END
)

REM 根据用户的选择，进入对应的目录并运行训练脚本
IF "%CHOICE%"=="1" (
    ECHO.
    ECHO --- Starting CLAP experiment ---
    cd /d "CLAP"
    python train.py
) ELSE IF "%CHOICE%"=="2" (
    ECHO.
    ECHO --- Starting Qwen2-Audio experiment ---
    cd /d "Qwen2_Audio"
    python train.py
) ELSE IF "%CHOICE%"=="3" (
    ECHO.
    ECHO --- Starting Audio_Flamingo_2 experiment ---
    cd /d "Audio_Flamingo_2"
    python train.py
) ELSE (
    ECHO.
    ECHO Invalid choice. Exiting.
)

:END
REM 训练结束后，保持命令行窗口打开，以便查看所有输出信息
ECHO.
ECHO --- Script finished. Press any key to exit. ---
pause >nul