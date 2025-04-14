FROM python:3.11.5

COPY ./ /BreakingArt/

# cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# install python dependencies
RUN pip install -r /BreakingArt/requirements.txt

WORKDIR /BreakingArt/
