FROM tensorflow/tensorflow
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install pillow
WORKDIR /app