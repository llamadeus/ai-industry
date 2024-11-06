# Install project dependencies using Poetry
install:
    poetry install

# Run the jupyterlab dev environment
lab: install
    cd notebooks && poetry run jupyter notebook
