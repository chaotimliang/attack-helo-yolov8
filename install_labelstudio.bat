@echo off
echo Installing Label Studio...
echo.

REM Try different pip commands
echo Trying pip3...
pip3 install label-studio
if %errorlevel% equ 0 goto success

echo Trying python -m pip...
python -m pip install label-studio
if %errorlevel% equ 0 goto success

echo Trying py -m pip...
py -m pip install label-studio
if %errorlevel% equ 0 goto success

echo Trying conda...
conda install -c conda-forge label-studio
if %errorlevel% equ 0 goto success

echo All installation methods failed. Please try manually:
echo 1. pip install label-studio
echo 2. conda install -c conda-forge label-studio
echo 3. Download from https://github.com/HumanSignal/label-studio
goto end

:success
echo.
echo Label Studio installed successfully!
echo.
echo Next steps:
echo 1. Initialize database: label-studio init
echo 2. Start server: label-studio start
echo 3. Open browser to http://localhost:8080
echo.

:end
pause
