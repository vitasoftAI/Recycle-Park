# Recycle-Park

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different used automobile parts (specifically, Genesis, KIA, and Hyundai). The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)). The model can be trained using two different frameworks ([PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/vitasoftAI/Recycle-Park.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Data Visualization

a) Genesis

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/156672f8-de64-49d0-9df5-caa606b5829a)

b) KIA

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/c10ff203-5d1a-47b5-8c28-e3828d2c4615)

c) Hyundai

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/122a346b-1d4f-4f52-9f80-c20f30e7f79a)

4. Train the AI model using the following PyTorch Lightning training script:

a) Genesis

```python
python train.py --data "genesis30_50" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/cb2b2dcc-0c58-4942-af65-0656aa0ea288)

b) KIA

```python
python train.py --data "new_kia" --batch_size = 64 devices = 4 --epochs 50
```

c) Hyundai

```python
python train.py --data "new_hyundai" --batch_size = 64 devices = 4 --epochs 50
```



![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/8a5c9cca-0083-4aa6-a488-80ce68414826)
![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/196483c9-c295-4af5-a417-881bd9106b4f)



