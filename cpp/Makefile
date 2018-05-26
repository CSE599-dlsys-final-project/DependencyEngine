.PHONY : clean

CXX = g++

CPPFLAGS += -std=c++14 -g -Wall -Wpedantic -O0
LDFLAGS= -shared

SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.hpp)
OBJECTS=$(SOURCES:.cpp=.o)

TARGET=dependencyengine.so

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
