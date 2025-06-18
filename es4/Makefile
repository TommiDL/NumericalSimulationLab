CC = g++
CFLAGS = -O3 --std=c++11
AFLAGS = -larmadillo

INCLUDE_DIRS = ../../NSL_SIMULATOR/SOURCE/


all: simulator.exe
simulator.exe : system.o NSL_SIMULATOR.o particle.o random.o
	$(CC) system.o particle.o random.o NSL_SIMULATOR.o -o simulator.exe $(AFLAGS)
NSL_SIMULATOR.o : NSL_SIMULATOR.cpp $(INCLUDE_DIRS)system.h $(INCLUDE_DIRS)particle.h
	$(CC) -c NSL_SIMULATOR.cpp -o NSL_SIMULATOR.o $(CFLAGS)
system.o : $(INCLUDE_DIRS)system.cpp $(INCLUDE_DIRS)system.h $(INCLUDE_DIRS)random.h $(INCLUDE_DIRS)particle.h
	$(CC) -c $(INCLUDE_DIRS)system.cpp -o system.o $(CFLAGS)
particle.o : $(INCLUDE_DIRS)particle.cpp $(INCLUDE_DIRS)particle.h
	$(CC) -c $(INCLUDE_DIRS)particle.cpp -o particle.o $(CFLAGS)
random.o : $(INCLUDE_DIRS)random.cpp $(INCLUDE_DIRS)random.h
	$(CC) -c $(INCLUDE_DIRS)random.cpp -o random.o $(CFLAGS)
clean :
	rm *.o
cleanall:
	rm *.o simulator.exe
remove:
	rm ../OUTPUT/*.*
	rm ../OUTPUT/CONFIG/*.*
