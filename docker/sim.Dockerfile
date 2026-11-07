FROM nvcr.io/nvidia/isaac-sim:6.2.0
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/rig
WORKDIR /rig
ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["/rig/rig/sim/main.py"]
