FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    LD_LIBRARY_PATH=/opt/conda/envs/chrombpnet-refactor/lib

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends ca-certificates git libcairo2 \
      libffi-dev libgdk-pixbuf2.0-0 libpango-1.0-0 libpangocairo-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/chrombpnet-refactor
COPY . .

RUN conda env create -f environment.yml && conda clean --all --yes

SHELL ["conda", "run", "--no-capture-output", "-n", "chrombpnet-refactor", "/bin/bash", "-c"]
RUN python -c "import torch; import chrombpnet; print(torch.__version__, chrombpnet.__version__)" && \
    chrombpnet --help >/dev/null

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "chrombpnet-refactor", "chrombpnet"]
CMD ["--help"]
