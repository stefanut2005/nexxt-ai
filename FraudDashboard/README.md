# FraudDashboard - Razor Pages frontend

This is a skeleton Razor Pages frontend for a Python fraud detection backend.

Run:
1. Start Python backend (FastAPI/Flask) on http://localhost:5001 with:
   - GET /transactions
   - POST /chat

2. dotnet restore
3. dotnet run

The app runs at http://localhost:5000 and proxies requests to the Python backend via /proxy/* endpoints.
