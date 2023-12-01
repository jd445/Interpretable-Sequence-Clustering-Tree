CC = g++
CFLAGS = -std=c++11 -Wall

all: RandomProjection

RandomProjection: RandomProjection.o
	$(CC) $(CFLAGS) -o RandomProjection RandomProjection.o

RandomProjection.o: RandomProjection.cpp
	$(CC) $(CFLAGS) -c RandomProjection.cpp

clean:
	rm -f *.o RandomProjection
