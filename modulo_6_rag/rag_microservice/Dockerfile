FROM python:3.11-slim
#
# install dependencies
COPY ./requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt
RUN pip install uvicorn
RUN pip install langchain_text_splitters
COPY . /usr/src/app
WORKDIR /usr/src/app