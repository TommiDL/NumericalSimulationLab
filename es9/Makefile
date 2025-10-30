CC = g++
CFLAGS = -O3 --std=c++11 -ggdb -fsanitize=address
AFLAGS = -larmadillo
OBJS = population.o map.o random.o operators.o function.o 
HEADS = random.h population.h operators.h map.h function.h

all: ga.exe test
test: test.o $(OBJS) $(HEADS)
	$(CC) $(CFLAGS) test.o $(OBJS) -o test $(AFLAGS)
ga.exe : es9.0.o $(OBJS) $(HEADS)
	$(CC) $(CFLAGS) es9.0.o $(OBJS) -o ga.exe $(AFLAGS)
%.o: %.cpp %.h 
	$(CC) $(CFLAGS) -c $<
clean :
	rm *.o