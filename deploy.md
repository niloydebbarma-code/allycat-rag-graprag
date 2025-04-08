# Deploying the Chat

**Table of contents**

- [1 - Build a Docker container](#1---build-a-docker-container)
    - [Checklist:](#checklist)
    - [Build Docker](#build-docker)
    - [Running the docker container locally](#running-the-docker-container-locally)
- [2 - Publishing Docker to DockerHub](#2---publishing-docker-to-dockerhub)
    - [Publishing to the dockerhub](#publishing-to-the-dockerhub)
    - [Running the DockerHub image](#running-the-dockerhub-image)
- [3 - Deploy to Google cloud](#3---deploy-the-application-to-google-cloud)


## 1 - Build a Docker container

### Checklist:

Run the following before building the docker container

- **Setup your local python env**.  [instructions](README.md#step-1-setup-python-env)
- **Setup .env file**. [instructions](README.md#61---setup-env-file-with-api-token)
- **Crawl the website** to download the content.  Run [1_crawl_site.ipynb](1_crawl_site.ipynb)
- **Process downloaded files**.  Run [2a_process_html_docling.ipynb](2a_process_html_docling.ipynb)  or [2b_process_html_dpk.ipynb](2b_process_html_dpk.ipynb)
- **Save to vector db**.  Run [3_save_to_vector_db.ipynb](3_save_to_vector_db.ipynb)
- **Query the data**.  Run [4_query.ipynb](4_query.ipynb)

### Build Docker

```bash
docker  build  -t allycat  .
```

Other docker options:

- ` --progress=plain`
- `--no-cache`

## Running the docker container locally

```bash
docker run -p 8080:8080 allycat
```

Go to URL:  http://localhost:8080


## 2 - Publishing Docker to DockerHub

### Publishing to the dockerhub

```bash
docker  login 

# tag the image to match dockerhub account
# Replace "USER" with your dockerhub username (e.g. 'sujee')
docker image tag  allycat    USER/allycat
# e.g.
docker image tag  allycat    sujee/allycat

# push it
docker  push   USER/allycat
# e.g.
docker  push   sujee/allycat
```

###  Running the DockerHub image

Once published, anyone can pull and run the docker image

```bash
docker run -p 8080:8080 sujee/allycat
```

Go to URL:  http://localhost:8080

## 3 - Deploy the application to Google Cloud

See [deploy-google-cloud-run.md](deploy-google-cloud-run.md)