import os
from ultralytics import YOLO

if __name__ == '__main__':

    # 设置数据集路径和训练参数
    data_yaml_path = './datasets/shipwreck/shipwreck.yaml'

    img_size = 640  # 输入图像大小
    epochs = 100  # 训练轮数
    batch = 16 
    verbose = True #在训练期间启用详细输出，提供详细的日志和进度更新。用于调试和密切监控训练过程。
    seed = 40 #设置用于训练的随机种子，确保结果在具有相同配置的运行中的可重复性。
    resume = True #从上次保存的检查点恢复训练。自动加载模型权重、优化器状态和 epoch 计数，无缝继续训练。
    plots = True #生成并保存训练和验证指标图以及预测示例，从而直观地了解模型性能和学习进度。
    lr0 = 0.01 #初始学习率 SGD
    lrf = 0.01 #最终学习率占初始学习率的分数 
    warmup_epochs = 3 #学习率预热的epochs数，逐渐将学习率从低值增加到初始学习率，以在早期稳定训练。
    warmup_momentum = 0.8 #预热阶段的初始动量，在预热期间逐渐调整到设定的动量。
    warmup_bias_lr = 0.1 #预热阶段偏差参数的学习率，有助于稳定初始时期的模型训练。
    iou = 0.7
    patience = 50
    
    model = YOLO("yolo11l.yaml").load('yolo11l.pt')
    # 开始训练
    results = model.train(data=data_yaml_path,
                          epochs=epochs,
                          imgsz=img_size,
                          batch=batch,
                          verbose=verbose,
                          seed=seed,
                          resume=resume,
                          plots=plots,
                          device = 2,
                          lr0 =lr0,
                          iou = iou,
                          patience = patience,
                          )


