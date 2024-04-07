CXX=g++
CXXFLAGS=-std=c++11 `pkg-config --cflags opencv4`
LIBS=`pkg-config --libs opencv4`

# Ziel für das Hauptprogramm
uebung1: main.o
	$(CXX) -o main main.o $(LIBS)

# Kompilieren der Hauptdatei
main.o: main.cpp
	$(CXX) -c main.cpp $(CXXFLAGS)
