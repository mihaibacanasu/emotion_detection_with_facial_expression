FROM tensorflow/tensorflow:latest
ENV PYTHONUNBUFFERED 1
RUn pip install --upgrade pip
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg 
RUN pip install pillow
# RUN SET OPENCV_LOG_LEVEL=debug
# RUN SET OPENCV_VIDEOIO_DEBUG=1
WORKDIR /app