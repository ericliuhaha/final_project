data_yaml_content = """
train: ./photo-label--1/train/images
val: ./photo-label--1/valid/images
test: ./photo-label--1/test/images

nc: 2
names: ['eyes', 'faces']

roboflow:
  license: CC BY 4.0
  project: photo-label
  url: https://universe.roboflow.com/bonktako/photo-label/dataset/1
  version: 1
  workspace: bonktako

"""

yaml_path = 'photo-label--1/data.yaml'
with open(yaml_path, 'w') as f:
    f.write(data_yaml_content)