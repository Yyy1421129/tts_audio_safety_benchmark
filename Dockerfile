FROM docker.v2.aispeech.com/sjtu/sjtu_yukai-yiyang_qwenaudio:v3

USER root

RUN sed -i 's@http://.*archive.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://.*security.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-cache-dir \
    librosa>=0.10.0 \
    scipy>=1.10.0 \
    soundfile>=0.12.0 \
    tqdm>=4.66.0 \
    pystoi>=0.4.1

USER rluser

CMD ["/bin/bash"]