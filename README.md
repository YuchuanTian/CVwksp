# Workshop: Introduction to Computer Vision
*UMJI-SSTIA Yuchuan Tian 12/2/2021*

## Config Environment
### Download link
https://jbox.sjtu.edu.cn/l/210aDv
Directory: CVwksp/software

### Install Visual Studio Code in Linux
- Download ```vscode.deb``` to your virtual machine;
- Open the directory where you store ```vscode.deb``` in terminal;
- Execute ```sudo dpkg -i vscode.deb```;
- Find the "Visual Studio Code" icon in "Show Applications" (Down left corner).
- Open the VScode application, and install any extension you like! (e.g. Python, Jupyter, Pylance, Markdown All in One...)

### Install Anaconda
- Download ```Anaconda3.sh``` to your virtual machine;
- Open the directory where you store ```Anaconda3.sh``` in terminal;
- Execute ```bash Anaconda3.sh```;
- Open a new terminal.

### Install Supporting Python Packages
- Open a new terminal;
- Execute ```conda activate base``` to activate your conda environment named "base";
- Execute the following commands:
```
pip install torch
pip install torchvision
pip install argparse
```
### Train Your First Neural Network!
- Open a terminal in this directory;
- Execute ```conda activate base```;
- No CUDA: execute ```python train.py --no_cuda```;
- With CUDA: execute ```sudo bitfusion run -n 1 -- /home/JI/ji51xxxxxxxxxx/anaconda3/bin/python train.py```.

### The "Pytorch Image Models" (```timm```) Library
- The ```timm``` library has various kinds of image classification models
- Link: https://rwightman.github.io/pytorch-image-models/
- Installation: ```pip install timm```.