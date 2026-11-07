FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends blender python3-numpy \
 && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["blender", "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1"]
