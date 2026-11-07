FROM nvcr.io/nvidia/isaac-sim:6.2.0
USER root
RUN mkdir -p /isaac-sim/.cache /isaac-sim/.nv/ComputeCache /isaac-sim/kit/cache \
        /isaac-sim/.nvidia-omniverse/logs /isaac-sim/.nvidia-omniverse/config /isaac-sim/.local/share/ov/data \
    && chown -R 1234:1234 /isaac-sim/.cache /isaac-sim/.nv /isaac-sim/kit/cache /isaac-sim/.nvidia-omniverse /isaac-sim/.local
USER 1234:1234
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/rig
WORKDIR /rig
ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["/rig/rig/sim/main.py"]
