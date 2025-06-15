CC = g++
CFLAGS= -Wall -O3 --std=c++11

RNDDIR=/home/tommaso/LSN/Parallel_number_generator_20250306/

LOCAL_EXECUTABLES:= es21

objects= func.o random.o random_walk.o

headers= ../es1/func.h $(RNDDIR)random.h random_walk.h

#all: $(objects) $(LOCAL_EXECUTABLES) 

#$(LOCAL_EXECUTABLES): %.o $(objects) $(header)
#	echo $@
#	echo $<

es211: es211.o $(objects)
	mkdir -p data
	$(CC) -o es211 es211.o $(objects) 

es212: es212.o $(objects)
	mkdir -p data
	$(CC) -o es212 es212.o $(objects) 

es221: es221.o $(objects)
	mkdir -p data
	$(CC) -o es221 es221.o $(objects) 

es222: es222.o $(objects)
	mkdir -p data
	$(CC) -o es222 es222.o $(objects) 

%.o: %.cpp $(header)
	$(CC) -c $< -o $@ $(CFLAGS)

func.o: ../es1/func.cpp ../es1/func.h
	$(CC) -c ../es1/func.cpp -o func.o $(CFLAGS)  	

random.o: $(RNDDIR)random.h $(RNDDIR)random.cpp
	$(CC) -c $(RNDDIR)random.cpp -o random.o

random_walk.o: random_walk.cpp random_walk.h $(objects) $(headers) 
	$(CC) -c random_walk.cpp -o random_walk.o


clean:
	rm *.o
