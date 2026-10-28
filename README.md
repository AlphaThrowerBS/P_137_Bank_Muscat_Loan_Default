🚀 [Loan Default Prediction System]

A full-stack machine learning application built to predict loan defaults, featuring robust ML tracking, a FastAPI backend, a Streamlit user interface, and containerized deployment with Docker.

---

## 🛠️ Tech Stack

* **Backend / API:** FastAPI, Python, Uvicorn
* **Machine Learning:** Scikit-learn, XGBoost, Random Forest, MLflow
* **Database:** PostgreSQL (Neon DB), SQLAlchemy
* **Frontend:** Streamlit
* **Containerization:** Docker & Docker Compose

---

## 📂 Project Structure

Main Folder
├── app/                   # Backend application code (FastAPI, database models, etc.)
├── frontend/              # Streamlit interface code
├── Dockerfile             # Container configuration for the backend/app
├── docker-compose.yml     # Multi-container Docker orchestration
├── dockerstar.bat         # Automated Windows startup batch script
├── requirements.txt       # Python dependencies
└── .env.example           # Template for environment variables

⚙️ Prerequisites
Before running the project, make sure you have the following installed on your machine:

Windows OS (Required to run the .bat script)

Docker Desktop (Must be installed and actively running in the background)

🚀 Getting Started
1. Clone the Repository
Open your terminal/command prompt and clone the repository

2. Configure Environment Variables
The application requires sensitive configuration (such as database URLs) which are not stored in version control for security reasons.

Rename the .env.example to .env
Open the.env file and fill in your actual credentials:
DATABASE_URL=your_postgresql_connection_string_here

3. Run the Project
You don't need to manually install Python or local dependencies (requirements.txt)—Docker handles everything inside isolated containers.

Simply double-click the run_docker.bat file in your project folder, or run it via Command Prompt:
run_docker.bat

🛑 Troubleshooting
sqlalchemy.exc.ArgumentError: Could not parse SQLAlchemy URL: This error means your container cannot read your database connection string. Ensure you have created a .env file in the root directory and that your docker-compose.yml includes the env_file: property pointing to .env.

Port Conflicts: If the app fails to start because a port (like 8000 or 8501) is already in use, stop any other local instances or database services running on your machine.