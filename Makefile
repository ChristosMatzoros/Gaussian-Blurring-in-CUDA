EXECUTABLES = pad_improved tiled tiled_streams

#This is the compiler to use
CC = nvcc 

CFLAGS =-Xcompiler -Wall -O4 -g


all: $(EXECUTABLES)
%: %.cu
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(EXECUTABLES)