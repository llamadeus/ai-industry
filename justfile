# Install project dependencies using Poetry
install:
    poetry install

# Run the jupyterlab dev environment
lab: install
    cd notebooks && poetry run jupyter notebook

# Run the jupyterlab within Docker
lab-docker:
    cd notebooks && poetry run jupyter notebook --allow-root --ip=0.0.0.0
