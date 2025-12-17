CC = mpicxx
CFLAGS = -O3 --std=c++11 -ggdb -fsanitize=address
AFLAGS = -larmadillo
OBJS = population.o map.o random.o operators.o function.o 
HEADS = random.h population.h operators.h map.h function.h

all: example1 example_bcast example_gather example_reduce example_commsplit example_unidircom example_timeperf
%: %.cpp
	$(CC) $< -o $@
clean :
	rm *.o