ARG PYTHON_VERSION=3.10

FROM python:$PYTHON_VERSION

# Switch work dir
WORKDIR /app

# Copy requirements.txt so that packages are not re-installed after every minor code change
COPY requirements.txt .

# Pip Installs
RUN pip3 install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org

# ENV variables
ENV PORT=5550

# Expose port
EXPOSE $PORT

CMD [ "flask", "run", "--host", "0.0.0.0:$PORT"]