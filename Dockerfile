# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables for uv and virtual environment

# https://andrich.me/2024/09/my-ideal-uv-based-dockerfile/
# Assure UTF-8 encoding is used.
ENV LC_CTYPE=C.utf8
# Location of the virtual environment
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
# Location of the python installation via uv
ENV UV_PYTHON_INSTALL_DIR="/python"
# Byte compile the python files on installation
ENV UV_COMPILE_BYTECODE=1
# Python verision to use
ENV UV_PYTHON=python3.12


# Create a non-root user
RUN useradd -m -u 1000 kurtis

# Set work directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tzdata cmake clang \
    && curl -LsSf https://astral.sh/uv/0.5.1/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tweaking the PATH variable for easier use
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"


# Change permissions for the /app directory
RUN chown -R kurtis:kurtis /app 

# Copy the pyproject.toml and poetry.lock files as the non-root user
COPY --chown=kurtis:kurtis pyproject.toml uv.lock /app/

# Switch to the non-root user
USER kurtis

# Install dependencies in a virtual environment inside the project
RUN uv sync --frozen --no-dev --no-install-project
 

# Copy the rest of the application code
COPY --chown=kurtis:kurtis . /app

# Run the CLI when the container starts
ENTRYPOINT ["uv", "run", "python3", "-m", "kurtis"]
