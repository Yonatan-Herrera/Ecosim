# EcoSim UI

This is a React-based UI for the EcoSim simulation.

## Setup

1.  **Backend**:
    The backend server is located in `../backend/server.py`.
    It requires `fastapi`, `uvicorn`, and `numpy`.
    
    Run the backend:
    ```bash
    cd ..
    uvicorn backend.server:app --reload --port 8002
    ```

2.  **Frontend**:
    The frontend is a Vite + React app.
    
    Run the frontend:
    ```bash
    npm install
    npm run dev
    ```

## Usage

-   Open the frontend URL (usually `http://localhost:5173`).
-   Click "EXECUTE" to start the simulation.
-   Go to "Config" to adjust parameters.
-   Go to "Logs" to see simulation events.
