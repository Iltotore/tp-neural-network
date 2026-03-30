rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

srcdir=src
libdir=lib
objdir=obj
bindir=bin
gendir=src-generated
objgendir=obj-generated
futdir=src-futhark

# Enable futhark/OpenCL compilation?
# Note: you might also need to enable/disable FUTHARK preprocessing variable in src/main.c
FUTHARK?=0

SRC:=$(call rwildcard,$(srcdir),*.c)
HEAD:=$(call rwildcard,$(srcdir),*.h)
OBJ:=$(subst $(srcdir), $(objdir), $(SRC:.c=.o))


PROG=$(bindir)/app

FUTSRC:=$(call rwildcard,$(futdir),*.fut)
OPENCL=-lOpenCL

ifeq ($(FUTHARK),1)
FUTC:=$(subst $(futdir),$(gendir),$(FUTSRC:.fut=.c))
FUTOBJ:=$(subst $(gendir),$(objgendir),$(FUTC:.c=.o))
OPENCL=-lOpenCL
else
FUTC=
FUTOBJ=
OPENCL=
endif

LIBS=-lmvec -lm $(OPENCL)

CC=gcc -W -Wall -Wextra -pedantic -std=c2x -O3 -march=native -ffast-math -DFUTHARK=$(FUTHARK)
RM=rm -ifR ./$(objdir)/* ./*.bak ./*.old ./$(bindir)/* ./$(gendir)/* ./$(objgendir)/*

all: $(PROG)

futhark: $(FUTC)

$(PROG): $(FUTOBJ) $(OBJ)
	@mkdir --parents $(bindir)
	@$(CC) $(OBJ) $(FUTOBJ) -o $(PROG) $(LIBS)

$(objdir)/%.o: $(srcdir)/%.c
	@mkdir --parents $(dir $@)
	@$(CC) -c $< -o $@

$(gendir)/%.c: $(futdir)/%.fut
	@mkdir --parents $(dir $@)
	@futhark opencl --library $< -o $(dir $@)/$(basename $(notdir $@))

$(objgendir)/%.o: $(gendir)/%.c
	@mkdir --parents $(dir $@)
	@$(CC) -c $< -o $@

clean:
	$(RM)

run: $(PROG)
	@$(PROG)