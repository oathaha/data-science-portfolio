# RAG with LlamaIndex


## Overview

This directory contains script to run chatbot for answering questions relating Australian visa. The chatbot is developed with streamlit. More details can be found in this [link](google-site).

The demo of this project can be seen in the below video (you need to click at the image to watch the video).

[![](https://img.youtube.com/vi/hwG7gCTKayw/0.jpg)](https://youtu.be/hwG7gCTKayw)

## Capability of the chatbot

The chatbot can answer the questions relating the following Australian visa:
- Permanent Residency visa
- Regional-Sponsored Migration visa
- Skilled Independent Point-Based Stream visa
- Skilled Nominated visa
- Skilled Work Regional visa
- Student visa
- Temporary Graduate visa (Post Higher Education Work Stream)
- Temporary Graduate visa (Post Vocational Education Work Stream)
- Training visa

## Data Collection

The details of the visa were collected from https://immi.homeaffairs.gov.au/visas/ on 4/Nov/2024.


## System Architecture

The details of the architecture are as follow:
- Vector store: Qdrant
- Sentence embedding model: all-mpnet-base-v2
- Response generator: GPT-4o


The data collection and ingestion pipeline is depicted in the below figure.

![](./figure/data-ingestion.png?raw=true "Data Collection and Ingestion Pipeline")

The steps for creating a vector store are as follows:

1. Crawl the webpages relating Australian visa by using FireCrawl. The crawled data are stored as markdown files.
2. Load markdown files as document objects and add metadata to them.
3. Parse the document objects into nodes and convert the contents in the nodes into embedding vectors.
4. Store the vectors in the Qdrant vector store.


![](./figure/chatbot-workflow.png?raw=true "Chatbot Workflow")

The workflow of the chatbot is as follows:

1. Convert a given query into an embedding vector by using a sentence embedding model.
2. Search for relevant documents in a vector store by using the embedding vector in step 1.
3. Post-process the obtained documents.
4. Create a prompt by using the post-processed documents and the given query.
5. Use GPT-4o to generate response from the prompt in step 4.

