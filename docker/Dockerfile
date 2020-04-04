FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install basic utilities
RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        build-essential \
        curl \
        cython3 \
        git \
        python-dev \
        python-pip \
        python-setuptools \
        python-wheel \
        python3-dev \
        python3-pip \
        python3-setuptools \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
ADD requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --requirement requirements.txt

# Download cocoapi
WORKDIR /opt
RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    git checkout 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9 && \
    sed -i -e 's|python|python3|g' Makefile && \
    make && \
    make install

# Fetch COCO dataset
RUN curl -O http://images.cocodataset.org/zips/train2014.zip && \
    unzip train2014.zip -d /opt/cocoapi/images && \
    rm train2014.zip

RUN curl -O http://images.cocodataset.org/zips/val2014.zip && \
    unzip val2014.zip -d /opt/cocoapi/images && \
    rm val2014.zip

RUN curl -O http://images.cocodataset.org/zips/test2014.zip && \
    unzip test2014.zip -d /opt/cocoapi/images && \
    rm test2014.zip

RUN curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip && \
    curl -O http://images.cocodataset.org/annotations/image_info_test2014.zip && \
    unzip annotations_trainval2014.zip -d /opt/cocoapi && \
    unzip image_info_test2014.zip -d /opt/cocoapi && \
    rm *2014.zip
