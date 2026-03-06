install:
	pip install --upgrade pip && \
		pip install -e .
	pip install tfrecord # Putting this here till we update the container.

editable-install:
	pip install --upgrade pip && \
		pip install -e .[dev] --config-settings editable_mode=strict

get-data:
	test -n "$(TEST_DATA_DIR)" || { echo "Error: TEST_DATA_DIR should be set"; exit 1; }
	mkdir -p $(TEST_DATA_DIR) && \
	rm -rf $(TEST_DATA_DIR)/modulus-data && \
	git clone https://gitlab-master.nvidia.com/modulus/modulus-data.git $(TEST_DATA_DIR)/modulus-data && \
	echo "Test data has been saved in ${TEST_DATA_DIR}"

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run ruff-format -a

interrogate:
	pre-commit run interrogate -a

lint:
	pre-commit run ruff-check -a && \
	pre-commit run markdownlint -a && \
	pre-commit run check-added-large-files -a

license: 
	python test/ci_tests/header_check.py --all-files

doctest:
	coverage run \
		--rcfile='test/coverage.docstring.rc' \
		-m pytest \
		--doctest-modules physicsnemo/ --ignore-glob=*internal* --ignore-glob=*experimental* --ignore-glob=*deploy/onnx*

pytest: 
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest --ignore-glob=*docs* --ignore-glob=*examples*

pytest-internal:
	cd test/internal && \
		pytest && \
		cd ../../

# NOTE: temporarily omitting diffusion coverage until we have a better way to test it.
coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --omit=*experimental* --omit=*diffusion* --fail-under=60 && \
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
else ifeq ($(ARCH), arm)
    TARGETPLATFORM := "linux/arm64"
else
    $(error Unknown CPU architecture ${ARCH} detected)
endif

PHYSICSNEMO_GIT_HASH = $(shell git rev-parse --short HEAD)

container-deploy:
	docker build -t physicsnemo:deploy --build-arg TARGETPLATFORM=${TARGETPLATFORM} --build-arg PHYSICSNEMO_GIT_HASH=${PHYSICSNEMO_GIT_HASH} --target deploy -f Dockerfile .

container-ci:
	docker build -t physicsnemo:ci --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target ci -f Dockerfile .

container-docs:
	docker build -t physicsnemo:docs --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target docs -f Dockerfile .

