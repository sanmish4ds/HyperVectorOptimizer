# Build: make PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config PG_SYSROOT=$(xcrun --show-sdk-path)
MODULE_big = hvo_pg
OBJS = hvo_pg.o
EXTENSION = hvo_pg
DATA = hvo_pg--1.0.sql
PG_CPPFLAGS = -I$(shell $(PG_CONFIG) --includedir-server)/extension -mcpu=native -O3
SHLIB_LINK = -framework Accelerate
PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
