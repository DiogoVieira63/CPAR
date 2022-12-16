CC = gcc
BIN= bin/
SRC = src/
EXEC = k_means
CFLAGS= -O3 -Wall -ftree-vectorize -mavx -fopenmp
SFLAGS =  -g -fno-omit-frame-pointer
.DEFAULT_GOAL = k_means
THREADS = 40


$(shell mkdir -p $(BIN))

k_means: $(SRC)k_means.c 
	$(CC) $(CFLAGS) $(SRC)k_means.c  -o $(BIN)$(EXEC)
clean: 
	rm -r bin/*

perfseq:
	 perf stat -e instructions,cycles,L1-dcache-load-misses -r 5 $(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

perfpar:
	 perf stat -e instructions,cycles,L1-dcache-load-misses -r 5 $(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)
	
runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)

	
assembly:
	$(CC) $(CFLAGS) $(SFLAGS) -S $(SRC)k_means.c  -o $(EXEC)
