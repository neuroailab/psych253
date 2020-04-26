run-tests:
	pytest

check-style:
	flake8 --show-source test_cv.py get_balanced_folds.py


# set to your username if you wish to push custom version to a different dockerhub acct
DOCKER_USERNAME = poldrack
current_dir = $(shell pwd)
backone = `dirname \`pwd\``

# get list of R packages needed by checkpoint
get-R-packages:
	cat *ipynb | grep "library(" | grep -v "Error" | sed 's/    "library(//' | sed 's/)\\n",//' | sort | uniq >| R_libraries.R

# code to check environment variables
# from https://stackoverflow.com/questions/4728810/makefile-variable-as-prerequisite

guard-%:
	@ if [ "${${*}}" = "" ]; then \
    	echo "Environment variable $* not set"; \
        exit 1; \
    fi
docker-build: guard-DOCKER_USERNAME
	docker build -t $(DOCKER_USERNAME)/jupyter-python-r .

docker-deploy: docker-login docker-upload

docker-login: guard-DOCKER_USERNAME guard-DOCKER_PASSWORD
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload: guard-DOCKER_USERNAME
	docker push $(DOCKER_USERNAME)/jupyter-python-r

shell: guard-DOCKER_USERNAME
	docker run -it -v $(current_dir):/analysis  $(DOCKER_USERNAME)/jupyter-python-r

jupyter: guard-DOCKER_USERNAME
	docker run -it -p 9991:9991 --entrypoint /opt/conda/bin/jupyter -v $(current_dir):/analysis $(DOCKER_USERNAME)/jupyter-python-r lab --ip=0.0.0.0 --port=9991 --allow-root
