set unstable # required for `[script]` tag instead of using shebang

silent := '>/dev/null 2>&1'

# Install project dependencies using Poetry
install:
    poetry install

# Run the jupyterlab dev environment
lab: install
    cd notebooks && poetry run jupyter notebook {{silent}} &

# Run the jupyterlab within Docker
lab-docker: install
    cd notebooks && poetry run jupyter notebook --allow-root --ip=0.0.0.0 {{silent}} &

[script]
kill-lab port="":
    if ! poetry run jupyter notebook stop {{port}}; then
        echo 'Please provide the port of the server you want to kill as parameter'
    fi
