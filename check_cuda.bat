@echo off
chcp 65001 >nul
echo Активация venv и проверка CUDA...
cd /d E:\CodeProjects\ModelsCompVision
call venv\Scripts\activate.bat
python -c "
import torch
print('=== PyTorch CUDA Information ===')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version in torch:', getattr(torch.version, 'cuda', 'None'))
print('cuDNN enabled:', torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else False)
if torch.backends.cudnn.enabled:
    print('cuDNN version:', torch.backends.cudnn.version())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name())
    print('Device capability:', torch.cuda.get_device_capability())
    print('Memory allocated:', torch.cuda.memory_allocated()/1024**3, 'GB')
    print('Memory cached:', torch.cuda.memory_reserved()/1024**3, 'GB')
else:
    print('CUDA devices: Not available')
"
pause





