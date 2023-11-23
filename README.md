## <div align="center">Documentation</div>

See below for a quickstart installation and usage example, and see the [YOLOv8 Docs](https://docs.ultralytics.com) for full documentation on training, validation, prediction and deployment.

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

 [Pytorch version Site] https://pytorch.org/get-started/previous-versions/
```bash
pip install -r requirements.txt

#CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>

<details open>
<summary>Usage</summary>

  #### CustomDataset
  
학습 전 셋팅이 필요한 yaml 파일들
```bash

(training hyperparameter) ./ultralytics/cfg/default.yaml

(dataset path) ./ultralytics/cfg/default.yaml 안에 data에서 customDataset.yaml 위치 지정

(참고로 data root path는 해당 위치에 지정되어 있음) ~/.config/Ultralytics/settings.yaml

(참고로 dataset 기본 양식 위치) ./ultralytics/models/yolo/detect/customDataset.yaml
```

#### Train File

```bash
./ultralytics/models/yolo/detect/train.py (폴더 최상단으로 이동)
```

#### predict setup
```bash
(training hyperparameter) ./ultralytics/cfg/default.yaml
(학습에 사용한 모델 settings 예시) model:  ./runs/detect/train19/weights/last.pt
(Prediction settings 에서 예측할 폴더 설정 예시 ) source:  './predict_dataset3'
```

#### Predict File
```bash
./ultralytics/models/yolo/detect/predict.py (폴더 최상단으로 이동)
```

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases). See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

</details>
