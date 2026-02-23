FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime


#
# install dependencies
COPY ./requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt --break-system-packages
COPY ./app /usr/src/app
WORKDIR /usr/src/app