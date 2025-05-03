from roboflow import Roboflow
rf = Roboflow(api_key="gCGD33xtms4MXZmmGpb2")
project = rf.workspace("bonktako").project("photo-label")
version = project.version(1)
dataset = version.download("yolov8")