from object_detection import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(__file__, model_label="yolov10", default_checkpoint="yolov10n.pt")
