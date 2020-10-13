FROM openjdk:8
FROM python:3.7

ENV PATH /opt/conda/bin:$PATH
ENV DISPLAY :0

# Run installers.
COPY installers /tmp/installers
RUN bash /tmp/installers/install_bazel.sh
RUN bash /tmp/installers/install_python_venv.sh

RUN pip3 install --upgrade pip
RUN pip3 install virtualenv==16.7.8
RUN pip3 install -r /tmp/installers/requirements.txt

# renewe bash
CMD [ "/bin/bash" ]
RUN mkdir -p /bark-ml
ENV WORKSPACE /bark-ml
ENV MPLBACKEND "agg"
WORKDIR /bark-ml