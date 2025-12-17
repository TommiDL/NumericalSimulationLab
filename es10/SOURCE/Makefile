CC = mpicxx
CFLAGS = -O3 --std=c++11 -ggdb #-fsanitize=address# -I/usr/lib64/mpich/include -L/usr/lib64/mpich/lib
AFLAGS = -larmadillo
OBJS = population.o map.o random.o operators.o function.o 
HEADS = random.h population.h operators.h map.h function.h

all: ga.exe
ga.exe : es10.o $(OBJS) $(HEADS)
	$(CC) $(CFLAGS) es10.o $(OBJS) -o ga.exe $(AFLAGS)
es10.o: es10.cpp $(HEADS)
	$(CC) $(CFLAGS) es10.cpp -c es10.o
%.o: %.cpp %.h 
	$(CC) $(CFLAGS) -c $<
clean :
	rm *.o