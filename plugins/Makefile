CC=g++
CFLAGS=-Weverything -shared -fPIC
CAFFE_TARGET=/home/alex/caffe_det
HEADERS=-I$(CAFFE_TARGET)/include -I/usr/local/cuda/include -I$(CAFFE_TARGET)/build/src
LDFLAGS=-L$(CAFFE_TARGET)/build/lib -L/usr/local/cuda/lib64
LIBS=-lglog -lboost_system -lboost_filesystem -lcudart -lcublas -lcurand -lcaffe

LAYERS=$(wildcard *.cpp)
OBJS=$(LAYERS:.cpp=.so)

all: $(OBJS)

%.so: %.cpp
	$(CC) -shared -fPIC $< -o $@ $(HEADERS) $(LDFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm *.so
