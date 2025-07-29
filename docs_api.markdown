# API Documentation

## Overview

The RAG Workflow API provides endpoints for querying academic papers using a Retrieval-Augmented Generation (RAG) workflow. It supports hybrid, HyDE, and step-back retrieval strategies, with authentication via API key and rate limiting for secure access.

## Authentication

All endpoints require an API key passed in the `X-API-Key` header. Set the `RAG_API_KEY` environment variable to configure the valid key.

## Endpoints

### GET /health
- **Description**: Check the health of the API and Ollama connection.
- **Response**:
  ```json
  {
    "status": "healthy",
    "ollama_available": true
  }
  ```
- **Rate Limit**: 10 requests per minute.

### POST /query
- **Description**: Submit a query to the RAG workflow.
- **Request Body**:
  ```json
  {
    "query": "string",
    "strategy": "hybrid|hyde|step_back"
  }
  ```
- **Response**:
  ```json
  {
    "result": "string",
    "status": "success",
    "query": "string",
    "strategy": "string"
  }
  ```
- **Rate Limit**: 5 requests per minute.

### GET /status
- **Description**: Get the status of the RAG workflow system.
- **Response**:
  ```json
  {
    "status": "operational",
    "pdf_count": 5,
    "index_available": true,
    "api_version": "1.0.0"
  }
  ```
- **Rate Limit**: 10 requests per minute.