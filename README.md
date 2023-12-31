# Recycle-Park

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different used automobile parts (specifically, Genesis, KIA, and Hyundai). The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)). The model can be trained using two different frameworks ([PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/vitasoftAI/Recycle-Park.git
```

2. Create conda environment from yml file using the following script:

a) Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n speed python=3.9
```

- Activate the environment using the following command:

```python
conda activate speed
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

b) Create a virtual environment using yml file:

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

Train process arguments can be changed based on the following information:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/df154acb-d49c-4096-84b7-7c1d632d6a19)

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

Training process progress:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/196483c9-c295-4af5-a417-881bd9106b4f)

c) Hyundai

```python
python train.py --data "new_hyundai" --batch_size = 64 devices = 4 --epochs 50
```

Training process progress:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/8a5c9cca-0083-4aa6-a488-80ce68414826)

5. Conduct inference using the trained model:
```python
python inference.py --data_name DATA_NAME --batch_size = 64 device = "cuda:0"
```

6. Demo using pretrained AI models:

a) Genesis

```python
python gradio_demo_gen.py
```

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/8480b0f9-ea14-468f-a58c-99ebb9cb6dee)

b) KIA

```python
python gradio_demo_kia.py
```

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/0143fa27-22d8-4d2f-adf6-92b13cbd826e)

c) Hyundai

```python
python gradio_demo_hy.py
```

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/b2c912ca-9b61-4fdc-9ac5-faaef60146cf)

7. Flask application:

a) Run the application in the terminal:

```python
python app.py
```

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/09001344-a4f4-493a-b2bd-d454de305828)

b) Go to the flask.ipynb file and run the cell:

![image](https://github.com/vitasoftAI/Recycle-Park/assets/50166164/aa2a3c1c-6485-4055-a2a4-8f2c2ad16cd1)
