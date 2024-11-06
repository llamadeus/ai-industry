FROM python:3.12.7

# Setup
WORKDIR /app
EXPOSE 8888

# This allows us to run `just`
ENV PATH="$PATH:/root/.local/bin"

# Install binaries
RUN apt update && \
    apt install -y pipx
RUN pipx ensurepath
RUN pipx install rust-just
RUN pip install poetry==1.8.3

# Install python dependencies
COPY justfile ./
COPY pyproject.toml poetry.lock ./
RUN just install

ENTRYPOINT ["just", "lab-docker"]
