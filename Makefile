CC = gcc
BIN= bin/
SRC = src/
EXEC = k_means
CFLAGS= -O2 -Wall -ftree-vectorize -mavx 
SFLAGS =  -g -fno-omit-frame-pointer
.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c 
	$(CC) $(CFLAGS) $(SRC)k_means.c  -o $(BIN)$(EXEC)
clean: 
	rm -r bin/*
run: 
	./$(BIN)$(EXEC)

assembly:
	$(CC) $(CFLAGS) $(SFLAGS) -S $(SRC)k_means.c  -o $(EXEC)

perf:
	 perf stat -e instructions,cycles,L1-dcache-load-misses  $(BIN)$(EXEC)
