# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies as root
RUN apt-get update && apt-get install -y 
RUN apt-get install build-essential -y
RUN apt-get install cmake -y
RUN apt-get install git -y
RUN apt-get install wget -y
RUN apt-get install libopencv-dev -y 
RUN apt-get install libwebsocketpp-dev -y 
RUN apt-get install nlohmann-json3-dev -y
RUN apt-get install libboost-all-dev -y
RUN apt-get install unzip -y

# Download and install Pylon SDK
RUN wget https://www2.baslerweb.com/media/downloads/software/pylon_software/pylon_7.3.0.27189_linux-x86_64_debs.tar.gz \
    && tar -xvzf pylon_7.3.0.27189_linux-x86_64_debs.tar.gz \
    && dpkg -i pylon_7.3.0.27189-deb0_amd64.deb \
    && rm pylon_7.3.0.27189-deb0_amd64.deb

# Download and install libtorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.4.1%2Bcpu.zip \
    && unzip libtorch-shared-with-deps-2.4.1+cpu.zip -d /usr/local \
    && rm libtorch-shared-with-deps-2.4.1+cpu.zip

# Testing https://www2.baslerweb.com/media/downloads/software/pylon_software/pylon_7.3.0.27189_linux-x86_64_debs.tar.gz

# Set environment variables for libtorch
ENV CMAKE_PREFIX_PATH=/usr/local/libtorch

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Build the project (assuming you have a CMakeLists.txt)
#RUN mkdir build && cd build && cmake .. && make

# Start shell instead
CMD ["/bin/bash"]

# Command to run your application
#CMD ["./build/real_time_monitoring"]
