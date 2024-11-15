set unstable # required for `[script]` tag instead of using shebang

silent := '>/dev/null 2>&1'

# Install project dependencies using Poetry
install:
    poetry install

setup: install
    poetry run nbdime config-git --enable # Enables better git-diffing for notebooks
    poetry run nbdime extensions --enable # Enables diffing in jupyterlab

# Run the jupyterlab dev environment
lab *options: install
    poetry run jupyter notebook --notebook-dir=notebooks {{options}} {{silent}} &

# Run the jupyterlab within Docker
lab-docker: (lab "--allow-root" "--ip=0.0.0.0")

[script]
kill-lab port="":
    if ! poetry run jupyter notebook stop {{port}}; then
        echo 'Please provide the port of the server you want to kill as parameter'
    fi

test: 
    poetry run pytest 'notebooks/util/util.py'
