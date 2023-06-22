# expected ubuntu dist for ShadeWatcher
FROM ubuntu:16.04

# commons
RUN apt update
RUN apt install -y software-properties-common git build-essential wget curl

# install g++8
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y g++-8 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# clone ShadeWatcher
RUN git clone https://github.com/syssec-utd/ShadeWatcher.git
ENV SHADEWATCHER_DIR=/ShadeWatcher

# >>> Parser Setup

# Setup ENV
ENV LIB_INSTALL_PATH=$SHADEWATCHER_DIR/lib

# neo4j lib
RUN add-apt-repository ppa:cleishm/neo4j && \
    apt-get update && \
    apt-get install -y libssl-dev neo4j-client libneo4j-client-dev


# libpqxx (c++ postgresql) prequisites
RUN apt install -y postgresql postgresql-contrib

# libpqxx (c++ postgresql)
#
# changed: 
#   git checkout master -> git checkout 7.6
# reason:
#   expected newer postgresql definitions (10+)
RUN apt install -y libpq-dev && \
    git clone https://github.com/jtv/libpqxx.git && \
    cd libpqxx && \
    git checkout 7.6 && \
    ./configure --disable-documentation --prefix=$LIB_INSTALL_PATH CXXFLAGS="-std=c++17" && \
    make -j8 && \
    make install && \
    cd .. && \
    rm -rf libpqxx

# libconfig
RUN wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.2.tar.gz && \
    tar xzvf libconfig-1.7.2.tar.gz && \
    cd libconfig-1.7.2/ && \
    ./configure --prefix=$LIB_INSTALL_PATH && \
    make -j8 && \
    make install && \
    cd .. && \
    rm -rf libconfig-1.7.2 libconfig-1.7.2.tar.gz

# jsoncpp
RUN apt install -y libjsoncpp-dev

# nlohmann json
RUN cd $LIB_INSTALL_PATH/include && \
    wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# librdkafka
RUN git clone https://github.com/edenhill/librdkafka.git && \
    cd librdkafka/ && \
    ./configure --prefix=$LIB_INSTALL_PATH && \
    make -j8 && \
    make install && \
    cd .. && \
    rm -rf librdkafka

# build deps
RUN apt install -y cmake pkg-config

# Setup system system library path (Optional)
ENV CPLUS_INCLUDE_PATH=$LIB_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH
ENV PATH=$LIB_INSTALL_PATH/bin:$PATH 
ENV LD_LIBRARY_PATH=$LIB_INSTALL_PATH/lib:$LIB_INSTALL_PATH/lib64:$LD_LIBRARY_PATH

# build and test
RUN cd $SHADEWATCHER_DIR/parse && \
    make -j8 && \
    ./driverdar -help && ./driverbeat -help 

# <<< Parser Setup

# >>> Recommendation Setup

# python 3.6
RUN add-apt-repository ppa:jblgf0/python && \
    apt update && \
    apt install -y python3.6

# pip 3.6
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3.6

# python deps
RUN pip3.6 install tensorflow_gpu==1.14
RUN pip3.6 install numpy==1.19.2
RUN pip3.6 install scipy==1.5.3
RUN pip3.6 install scikit-learn==0.19.0
RUN pip3.6 install tqdm colorlog

# test
RUN cd $SHADEWATCHER_DIR/recommend && \
    python3.6 driver.py -h

# <<< Recommendation Setup

# cleanup packages
RUN apt autoremove

ENTRYPOINT ["/bin/bash"]