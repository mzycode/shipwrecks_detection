
from ultralytics import YOLO,RTDETR


if __name__ == '__main__':

    model = YOLO("./runs/train/weights/best.pt")
    source = "./exsample"

    # Run inference on the source
    model.predict(
        source=source, 
        save=True,
        device=0,
        visualize = True,
        
        )  
