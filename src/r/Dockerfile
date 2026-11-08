FROM r-base:4.4.3

WORKDIR /app

COPY src/r/install_packages.R ./install_packages.R
RUN Rscript install_packages.R

COPY src/r/main.R ./main.R
ENV TITANIC_DATA_DIR=/app/data

CMD ["Rscript", "main.R"]
