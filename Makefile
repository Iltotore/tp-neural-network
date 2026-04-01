rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

srcdir=src
libdir=lib
objdir=obj
bindir=bin
gendir=src-generated
objgendir=obj-generated
futdir=src-futhark
benchdir=bench

# Enable futhark/OpenCL compilation?
# Note: you might also need to enable/disable FUTHARK preprocessing variable in src/main.c
FUTHARK?=0

SRC:=$(call rwildcard,$(srcdir),*.c)
HEAD:=$(call rwildcard,$(srcdir),*.h)
OBJ:=$(subst $(srcdir), $(objdir), $(SRC:.c=.o))


PROG=$(bindir)/app

FUTSRC:=$(call rwildcard,$(futdir),*.fut)
FUTBENCH:=$(subst $(futdir),$(benchdir),$(FUTSRC:.fut=.prof))
OPENCL=-lOpenCL

ifdef FUTHARK_DEVICE
FUTHARK=1
FUTFLAGS=-DFUTHARK=1 -DFUTHARK_DEVICE="\"$(FUTHARK_DEVICE)\""
else
FUTFLAGS=-DFUTHARK=$(FUTHARK)
endif

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

CC=gcc -W -Wall -Wextra -pedantic -std=c2x -O3 -march=native -ffast-math $(FUTFLAGS)
RM=rm -ifR ./$(objdir)/* ./*.bak ./*.old ./$(bindir)/* ./$(gendir)/* ./$(objgendir)/* ./$(benchdir)/*

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

$(benchdir)/%.fut: $(futdir)/%.fut
	@mkdir --parents $(dir $@)
	@cp $< $@

$(benchdir)/%.json: $(benchdir)/%.fut
	@futhark bench --backend=opencl $< --profile --json=$@

$(benchdir)/%.prof: $(benchdir)/%.json
	@futhark profile $<
	@mv $(basename $(notdir $@)).prof $@

benchmark: $(FUTBENCH)