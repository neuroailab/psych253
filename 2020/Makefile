run-tests:
	pytest

check-style:
	flake8 --show-source test_cv.py get_balanced_folds.py


# set to your username if you wish to push custom version to a different dockerhub acct
DOCKER_USERNAME = poldrack
current_dir = $(shell pwd)

# code to check environment variables
# from https://stackoverflow.com/questions/4728810/makefile-variable-as-prerequisite

guard-%:
	@ if [ "${${*}}" = "" ]; then \
    	echo "Environment variable $* not set"; \
        exit 1; \
    fi
docker-build: guard-DOCKER_USERNAME
	docker build -t $(DOCKER_USERNAME)/psych253 .

docker-deploy: docker-login docker-upload

docker-login: guard-DOCKER_USERNAME guard-DOCKER_PASSWORD
	docker login --username=$(DOCKER_USERNAME) --password=$(DOCKER_PASSWORD)

docker-upload: guard-DOCKER_USERNAME
	docker push $(DOCKER_USERNAME)/narps-analysis

docker-run: guard-DOCKER_USERNAME
	    docker run -it -p 9993:9993 -v $(current_dir):/analysis  $(DOCKER_USERNAME)/psych253 