set unstable # required for `[script]` tag instead of using shebang

silent := '>/dev/null 2>&1'

# Install project dependencies using Poetry
install:
    poetry install

setup: install
    poetry run nbdime config-git --enable # Enables better git-diffing for notebooks
    poetry run nbdime extensions --enable # Enables diffing in jupyterlab

# Run the jupyterlab dev environment
lab *options:
    # Only start new instance if none is running
    test -z "$(poetry run jupyter notebook list --json)" \
        && poetry run jupyter notebook --notebook-dir=notebooks {{options}} {{silent}} &

# Run the jupyterlab within Docker
lab-docker: (lab "--allow-root" "--ip=0.0.0.0")

[script]
kill-lab port="":
    if ! poetry run jupyter notebook stop {{port}}; then
        echo 'Please provide the port of the server you want to kill as parameter'
    fi

# Choose a file to open in the running lab instance
[script("bash")]
choose-notebook browser="open": lab
    choice=$(find notebooks/ -type f -name "*.ipynb" -not -path "*/.*/*" | fzf --sort) # List notebooks ignoring hidden folders
    eval $(poetry run jupyter notebook list --json | jq -r '"token=\(.token) url=\(.url)"') # Get token and url of server
    http_encoded_choice=${choice// /%20}
    {{browser}} "${url}${http_encoded_choice}?token=${token}"

test: 
    poetry run pytest 'notebooks/util/util.py'
