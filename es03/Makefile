CC = g++
CFLAGS= -Wall -O3 --std=c++11

RNDDIR=/home/tommaso/LSN/Parallel_number_generator_20250306/

LOCAL_EXECUTABLES:= es301 es302

objects= func.o random.o 

headers= ../es1/func.h $(RNDDIR)random.h 

#all: $(objects) $(LOCAL_EXECUTABLES) 

#$(LOCAL_EXECUTABLES): %.o $(objects) $(header)
#	echo $@
#	echo $<

es301: es301.o $(objects)
	mkdir -p data
	$(CC) -o es301 es301.o $(objects) 


es302: es302.o $(objects)
	mkdir -p data
	$(CC) -o es302 es302.o $(objects) 

%.o: %.cpp $(header)
	$(CC) -c $< -o $@ $(CFLAGS)

func.o: ../es1/func.cpp ../es1/func.h
	$(CC) -c ../es1/func.cpp -o func.o $(CFLAGS)  	

random.o: $(RNDDIR)random.h $(RNDDIR)random.cpp
	$(CC) -c $(RNDDIR)random.cpp -o random.o


clean:
	rm *.o
