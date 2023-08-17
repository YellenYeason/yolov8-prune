from ultralytics import YOLO  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.train(data="ultralytics/cfg/datasets/VOC.yaml", epochs=3,batch=32,workers=8)