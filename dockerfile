# One image per engine environment, envs/chrombpnet-<engine>.yml:
#
#   docker build --build-arg ENGINE=pytorch    -t chrombpnet-refactor:pytorch .
#   docker build --build-arg ENGINE=jax        -t chrombpnet-refactor:jax .
#   docker build --build-arg ENGINE=tensorflow -t chrombpnet-refactor:tensorflow .
#
#   docker run --rm --gpus all -v "$PWD":/data -w /data chrombpnet-refactor:pytorch pred_bw ...
#   apptainer build chrombpnet-pytorch.sif docker-daemon://chrombpnet-refactor:pytorch
#   apptainer exec --nv chrombpnet-pytorch.sif chrombpnet pred_bw ...
#
# The tensorflow image runs both TensorFlow engines (--engine tensorflow, the
# default there, and --engine tensorflow-old). Each environment brings its own
# CUDA libraries with its framework's wheels, so the base is a plain conda
# image; the host needs only an NVIDIA driver new enough for them
# (envs/README.md), and Ubuntu 24.04's glibc suits every wheel, including on
# hosts whose own glibc is too old for them.
FROM condaforge/miniforge3:26.7.2-0

ARG ENGINE=pytorch

# The environment is on PATH rather than activated, so `apptainer exec image
# chrombpnet ...` finds it too. Apptainer/Singularity passes the host's
# environment into the container, and with it the user's ~/.local
# site-packages and any PYTHONPATH/PYTHONHOME; the image's own values below
# win, so host packages cannot shadow the image's. LD_LIBRARY_PATH holds the
# environment's own libraries only (pango, cairo for the PDF reports), never a
# system CUDA that would shadow the wheels'.
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CHROMBPNET_IMAGE_ENGINE=${ENGINE} \
    CONDA_PREFIX=/opt/conda/envs/chrombpnet-${ENGINE} \
    CONDA_DEFAULT_ENV=chrombpnet-${ENGINE} \
    PATH=/opt/conda/envs/chrombpnet-${ENGINE}/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/conda/envs/chrombpnet-${ENGINE}/lib \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="" \
    PYTHONHOME=""

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends ca-certificates git libcairo2 \
      libffi-dev libgdk-pixbuf-2.0-0 libpango-1.0-0 libpangocairo-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/chrombpnet
COPY . .

# Then every file readable by any user: Apptainer runs the image as the
# host's user, and COPY kept the checkout's own modes (often group-only).
RUN test -f "envs/chrombpnet-${ENGINE}.yml" || { \
      echo "ENGINE must be pytorch, jax or tensorflow (envs/chrombpnet-<engine>.yml), not '${ENGINE}'" >&2; exit 1; } && \
    conda env create -f "envs/chrombpnet-${ENGINE}.yml" && conda clean --all --yes && \
    chmod -R a+rX /opt/chrombpnet

# The image's engine imports, and the CLI starts.
RUN python -c "import chrombpnet, chrombpnet.engines as e; e.get('${ENGINE}').require(); print('chrombpnet', chrombpnet.__version__, 'engine ${ENGINE}')" && \
    chrombpnet --help >/dev/null

ENTRYPOINT ["chrombpnet"]
CMD ["--help"]
