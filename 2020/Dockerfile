### Dockerfile for Jupyter environment with R

FROM jupyter/datascience-notebook

WORKDIR /analysis

RUN echo 'install.packages(c("lmerTest", "optimx"), repos="http://cran.us.r-project.org", dependencies=TRUE)' >> /tmp/packages.R && Rscript /tmp/packages.R

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--port 9993","--allow-root"]
