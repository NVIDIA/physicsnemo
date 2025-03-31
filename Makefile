install:
	pip install --upgrade pip && \
		pip install -e .

editable-install:
	pip install --upgrade pip && \
		pip install -e .[dev] --config-settings editable_mode=strict

get-data:
	mkdir -p /data && \
		mkdir -p /data/nfs/ && \
		git -C /data/nfs/modulus-data pull || \
		git clone https://gitlab-master.nvidia.com/modulus/modulus-data.git /data/nfs/modulus-data

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	pre-commit run interrogate -a

lint:
	pre-commit run markdownlint -a && \
	pre-commit run ruff -a && \
	pre-commit run check-added-large-files -a

license: 
	pre-commit run license -a

doctest:
	coverage run \
		--rcfile='test/coverage.docstring.rc' \
		-m pytest \
		--doctest-modules physicsnemo/ --ignore-glob=*internal* --ignore-glob=*experimental*

pytest: 
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest --ignore-glob=*docs* --ignore-glob=*examples*

pytest-internal:
	cd test/internal && \
		pytest && \
		cd ../../

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --omit=*experimental* --fail-under=70 && \
		coverage html

all-ci: get-data setup-ci black interrogate lint license install pytest doctest coverage

# For arch naming conventions, refer
# https://docs.docker.com/build/building/multi-platform/
# https://github.com/containerd/containerd/blob/v1.4.3/platforms/platforms.go#L86
ARCH := $(shell uname -p)

ifeq ($(ARCH), x86_64)
    TARGETPLATFORM := "linux/amd64"
else ifeq ($(ARCH), aarch64)
    TARGETPLATFORM := "linux/arm64"
else
    $(error Unknown CPU architecture ${ARCH} detected)
endif

MODULUS_GIT_HASH = $(shell git rev-parse --short HEAD)

container-deploy:
	docker build -t physicsnemo:deploy --build-arg TARGETPLATFORM=${TARGETPLATFORM} --build-arg MODULUS_GIT_HASH=${MODULUS_GIT_HASH} --target deploy -f Dockerfile .

container-ci:
	docker build -t physicsnemo:ci --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target ci -f Dockerfile .

container-docs:
	docker build -t physicsnemo:docs --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target docs -f Dockerfile .

