@echo off
chcp 65001 >nul
title 逐字 Verbatim 本機引擎安裝
setlocal
set "VB=%LOCALAPPDATA%\Verbatim"
if not exist "%VB%" mkdir "%VB%"
set "UV=%VB%\uv\uv.exe"
if exist "%UV%" goto haveuv
echo [準備] 下載安裝工具 uv ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%VB%\uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/0.11.32/install.ps1 | iex"
if exist "%VB%\uv\bin\uv.exe" set "UV=%VB%\uv\bin\uv.exe"
if exist "%UV%" goto haveuv
echo 下載 uv 失敗，請確認網路連線後再執行一次。
goto fail

:haveuv
if defined VERBATIM_LAUNCH_SRC goto copylaunch
curl -fsSL -o "%VB%\launch.py" "https://raw.githubusercontent.com/bevis0414-crypto/verbatim/stable/local/launch.py"
if errorlevel 1 goto dlfail
goto run

:copylaunch
copy /y "%VERBATIM_LAUNCH_SRC%" "%VB%\launch.py" >nul
goto run

:dlfail
echo 下載啟動器失敗，請確認網路連線後再執行一次。
goto fail

:run
set "PYTHONIOENCODING=utf-8"
set "VERBATIM_UV=%UV%"
"%UV%" run --no-project --python 3.11 "%VB%\launch.py" --install %*
set "RC=%errorlevel%"
echo.
echo 這個視窗可以關掉了。
if not defined CI pause
exit /b %RC%

:fail
if not defined CI pause
exit /b 1
