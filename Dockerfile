#下载基础镜像
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
# LABEL 备注信息
LABEL MAINTAINER="fzy"
LABEL version="1.0"
LABEL description="comModel"
# 非交互模式
#ENV DEBIAN_FRONTEND noninteractive

#
RUN \
    DEBIAN_FRONTEND=noninteractive apt-get update &&  \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip git openssh-server curl locales clang &&  \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyyaml pymysql django django-tables2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch accelerate protobuf datasets "chardet<3.1.0" "urllib3<=1.25" "sentencepiece<0.1.92" sklearn transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN sed -ie 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/g' /etc/locale.gen
RUN locale-gen
ENV LANG zh_CN.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HOME /root

# 安装node
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - &&  \
    DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs

RUN npm install nyc jsvu jshint estraverse esprima escodegen commander -g --registry=http://registry.npmmirror.com
ENV NODE_PATH /usr/lib/node_modules/
ENV COMMODEL /root/comModel
LABEL key="value49"

#拷贝引擎
ADD dataset/jsvu.tar.gz /root/

WORKDIR $COMMODEL

#开启ssh服务
RUN mkdir /var/run/sshd
RUN echo 'root:123456' | chpasswd
RUN echo "Port 10343" >> /etc/ssh/sshd_config
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo "service ssh restart" >> ~/.bashrc