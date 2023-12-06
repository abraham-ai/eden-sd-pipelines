@echo off

:: Activate an existing virtual environment
call venv\Scripts\activate

:: Install the desired packages including Torch
pip3 install clean-fid numba numpy torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118

:: Rest of your installation script goes here
:: ...

:: Deactivate the virtual environment when done (optional)
deactivate
