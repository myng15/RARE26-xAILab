FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime AS rare26-dinov2-lora-attnmil-ensemble-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
# DINOv2 uses a plain-attention fallback when xFormers is absent (deterministic, no build step).
ENV XFORMERS_DISABLED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user third_party /opt/app/third_party
COPY --chown=user:user model /opt/app/model
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
