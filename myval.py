
from ultralytics import YOLO

import torch
if __name__ == '__main__':

    data_yaml_path = '.datasets/shipwreck/shipwreck.yaml'
    # Load a model
    model = YOLO("./runs/train/weights/best.pt")

    # Customize validation settings
    validation_results = model.val(data=data_yaml_path,
                                   imgsz=640,
                                   batch=16,
                                #    split='test', #是否应用测试集
                                   device= 0,
                                   plots=True,

                                   )

