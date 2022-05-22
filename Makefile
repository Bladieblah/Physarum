PROGNAME = a.out
SRCDIR = src/
OBJDIR = obj/

SRC	= $(wildcard $(SRCDIR)*.cpp)

CC = g++ -std=c++17

WARNFLAGS = -Wall -Wno-deprecated-declarations
CFLAGS = -g -O3 $(WARNFLAGS) -Iinclude/ -I./ -I/usr/local/include

LDFLAGS =-framework opencl -framework GLUT -framework OpenGL -framework Cocoa -L/usr/local/lib -lGLEW

# Do some substitution to get a list of .o files from the given .cpp files.
OBJFILES = $(patsubst $(SRCDIR)%.cpp, $(OBJDIR)%.o, $(SRC))

.PHONY: all clean

all: $(PROGNAME)

$(PROGNAME): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -fv $(PROGNAME) $(OBJFILES)
