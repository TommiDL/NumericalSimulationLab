CC = g++
CFLAGS= -Wall -O3 --std=c++11 -g

RNDDIR=/home/tommaso/LSN/Parallel_number_generator_20250306/

LOCAL_EXECUTABLES:= es5

objects= func.o random.o probability.o

headers= ../es1/func.h $(RNDDIR)random.h probability.h

#all: $(objects) $(LOCAL_EXECUTABLES) 

#$(LOCAL_EXECUTABLES): %.o $(objects) $(header)
#	echo $@
#	echo $<

es5: es5.o $(objects)
	mkdir -p data
	$(CC)  -fsanitize=address -o es5 es5.o $(objects) 

%.o: %.cpp $(header)
	$(CC) -c $< -o $@ $(CFLAGS)

func.o: ../es1/func.cpp ../es1/func.h
	$(CC) -c ../es1/func.cpp -o func.o $(CFLAGS)  	

random.o: $(RNDDIR)random.h $(RNDDIR)random.cpp
	$(CC) -c $(RNDDIR)random.cpp -o random.o


clean:
	rm *.o
