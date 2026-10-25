@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag one or more photos onto this .bat file.
  echo Please drag photos onto this file instead of double-clicking it.
  pause
  exit /b 1
)
if not exist output mkdir output
set "UP="
if exist "weights\upscale\anime.onnx" set "UP=--upscale anime"
:loop
if "%~1"=="" goto done
echo Processing %~nx1 ...
ptoart.exe convert "%~1" "output\%~n1-avatar.png" --size 1024 %UP% --model-dir weights --device cpu --overwrite
shift
goto loop
:done
echo.
echo Done. Results are in the "output" folder.
start "" "%~dp0output"
pause
