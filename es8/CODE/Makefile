CC = g++
CFLAGS = -O3 --std=c++11
AFLAGS = -larmadillo
UTILS  = function.h

all: simulator.exe
simulator.exe : system.o annealing.o random.o 
	$(CC) system.o random.o annealing.o -o simulator.exe $(AFLAGS)
annealing.o : annealing.cpp system.h $(UTILS)
	$(CC) -c annealing.cpp -o annealing.o $(CFLAGS) 
system.o : system.cpp system.h random.h $(UTILS)
	$(CC) -c system.cpp -o system.o $(CFLAGS)
random.o : random.cpp random.h $(UTILS)
	$(CC) -c random.cpp -o random.o $(CFLAGS)
clean :
	rm *.o
cleanall :
	rm *.o simulator.exe