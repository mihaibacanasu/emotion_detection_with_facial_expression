FROM tensorflow/tensorflow:latest

ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg
RUN apt-get install -y v4l-utils
RUN pip install pillow
RUN pip install matplotlib

WORKDIR /app
COPY display.py /app/display.py

CMD ["python", "display.py"]