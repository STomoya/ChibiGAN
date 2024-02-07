# find docker-compose v1
# if not, suppose we have docker-compose v2
_v1_bin := $(shell command -v docker-compose 2> /dev/null)
ifdef _v1_bin
COMPOSE_COMMAND := "docker-compose"
else
COMPOSE_COMMAND := "docker compose"
endif

run:
	${COMPOSE_COMMAND} run --rm torch python main.py ${args}
drun:
	${COMPOSE_COMMAND} run --rm -d torch python main.py ${args}
