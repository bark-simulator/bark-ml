FROM openjdk:8
FROM python:3.7
FROM tensorflow/tensorflow:2.2.0-gpu

ENV PATH /opt/conda/bin:$PATH
ENV DISPLAY :0
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
python3.7 \
git

# Run installers.
COPY installers /tmp/installers
RUN bash /tmp/installers/install_bazel.sh
RUN bash /tmp/installers/install_python_venv.sh

RUN pip3 install --upgrade pip
RUN pip3 install virtualenv
RUN pip3 install -r /tmp/installers/requirements.txt

# renewe bash
CMD [ "/bin/bash" ]
WORKDIR /bark
ENV DEBIAN_FRONTEND interactive
