### Dockerfile for Jupyter environment with R

FROM jupyter/datascience-notebook

WORKDIR /analysis

USER root

RUN apt-get update && apt-get install -y --no-install-recommends libv8-dev graphviz \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/ijmbarr/causalgraphicalmodels
RUN pip install brainnetworks python-igraph biopython

RUN echo 'install.packages(c("semPlot", "fabricatr", "EMAtools", "psych", "simr", "RColorBrewer", "brms", "lavaan", "lavaanPlot", "lmerTest", "optimx", "semTools"), repos="http://cran.us.r-project.org", dependencies=TRUE)' >> /tmp/packages.R && Rscript /tmp/packages.R

RUN pip install pytest wget skggm 
RUN apt-get update && apt-get install -y --no-install-recommends mysql-client libmariadbclient-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo 'install.packages(c("forecast", "wordbankr"), repos="http://cran.us.r-project.org", dependencies=TRUE)' >> /tmp/packages.R && Rscript /tmp/packages.R

ENTRYPOINT ["/bin/bash"] 
