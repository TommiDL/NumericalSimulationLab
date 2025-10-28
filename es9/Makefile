CC = g++
CFLAGS = -O3 --std=c++11 -ggdb
AFLAGS = -larmadillo
OBJS = es9.0.o population.o map.o random.o operators.o function.o
HEADS = random.h population.h operators.h map.h function.h

all: ga.exe
ga.exe : $(OBJS) $(HEADS)
	$(CC) $(OBJS) -o ga.exe $(AFLAGS)
%.o: %.cpp %.h 
	$(CC) $(CFLAGS) -c $<
clean :
	rm *.o ga.exe