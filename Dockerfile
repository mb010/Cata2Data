FROM python:3.9

RUN pip install git+https://github.com/miguelcarcamov/cata2data@main#egg=mypackage

RUN echo "Hello from cata2data docker image"
LABEL org.opencontainers.image.source="https://github.com/mb010/cata2data"
LABEL org.opencontainers.image.description="Container image for the cata2data package"
LABEL org.opencontainers.image.licenses=GPL3
