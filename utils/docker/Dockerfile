FROM openjdk:8
FROM python:3.7

ENV PATH /opt/conda/bin:$PATH
ENV DISPLAY :0


# Run installers.
COPY installers /tmp/installers
COPY requirements.txt /tmp/requirements.txt
RUN bash /tmp/installers/install_bazel.sh
RUN bash /tmp/installers/install_python_venv.sh

# upgrade pip before creating virtualenv
RUN pip3 install --upgrade pip
RUN pip3 install virtualenv
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install tensorflow-gpu==2.1.0


RUN apt-get update

# renewe bash
CMD [ "/bin/bash" ]
WORKDIR /bark