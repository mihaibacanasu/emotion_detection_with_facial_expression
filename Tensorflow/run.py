import os

os.system('docker build -t emot .')
os.system(f'docker run --rm -it -v {os.path.dirname(os.path.abspath(__file__))}:/app emot python train.py')