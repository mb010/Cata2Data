FROM python:3.9

RUN pip install git+https://github.com/miguelcarcamov/cata2data@main#egg=mypackage

RUN echo "Hello from cata2data docker image"
LABEL org.opencontainers.image.source="https://github.com/mb010/Cata2Data"
LABEL org.opencontainers.image.description="Docker container image for the cata2data package"
LABEL org.opencontainers.image.licenses=GPL3
