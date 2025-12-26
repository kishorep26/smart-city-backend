# Sentinel Command Backend API

## Overview
A robust FastAPI-based backend service for the Sentinel Command security monitoring system. This API provides comprehensive incident management, agent tracking, and real-time metrics for security operations. Built with Python and SQLAlchemy, it offers scalable database management and efficient API endpoints for frontend integration.

## Key Features
- RESTful API built with FastAPI framework
- Incident management and tracking system
- Agent registration and monitoring
- Real-time statistics and metrics endpoints
- SQLAlchemy ORM for database operations
- CORS middleware for cross-origin requests
- Database session management and connection pooling
- Comprehensive error handling and validation
- Modular API structure with organized routes

## Technology Stack
- Backend Framework: Python, FastAPI
- ORM: SQLAlchemy, Pydantic
- Database: SQLite/PostgreSQL compatible
- HTTP Client: httpx, httpcore
- API Documentation: Swagger/OpenAPI (auto-generated)
- Deployment: Vercel-compatible configuration

## Getting Started
1. Install dependencies: pip install -r requirements.txt
2. Set up environment variables for database connection
3. Initialize database: python database.py (creates tables)
4. Run the server: uvicorn main:app --reload
5. Access API documentation at http://localhost:8000/docs
6. Test API endpoints using Swagger UI or Postman

## Deployment
Configured for deployment on Vercel with Python runtime. Also compatible with AWS Lambda, Google Cloud Run, or any Python hosting platform.
