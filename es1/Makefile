CC = g++
CFLAGS= -Wall -O3 --std=c++11

RNDDIR=../Parallel_number_generator_20250306/

LOCAL_EXECUTABLES:= es101 es102 es103 es120

objects= func.o random.o rndgenmod.o

headers=func.h $(RNDDIR)random.h rndgendmod.h

#all: $(objects) $(LOCAL_EXECUTABLES) 

#$(LOCAL_EXECUTABLES): %.o $(objects) $(header)
#	echo $@
#	echo $<

es103: es103.o $(objects)
	$(CC) -o es103 es103.o $(objects) 

es120: es120.o $(objects)
	$(CC) -o es120 es120.o $(objects) 

%.o: %.cpp $(header)
	$(CC) -c $< -o $@ $(CFLAGS)

func.o: func.cpp func.h
	$(CC) -c func.cpp -o func.o $(CFLAGS)  	

random.o: $(RNDDIR)random.h $(RNDDIR)random.cpp
	$(CC) -c $(RNDDIR)random.cpp -o random.o
rndgenmod.o: rndgenmod.h rndgenmod.cpp $(RNDDIR)random.h
	$(CC) -c rndgenmod.cpp rndgenmod.h

clean:
	rm *.o
