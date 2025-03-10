# Depression Prediction Web App

This is a Streamlit-based web application that predicts the probability of depression in primary care patients based on input features. The app allows users to input various patient details and displays a predicted depression risk score.

## Prerequisites

Before running the app, make sure you have the following installed:

- **Python** (preferably version 3.7+)
- **Streamlit** (for web interface)
- **Required Libraries** (listed in `requirements.txt`)

## Setup Instructions

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://your-repository-url.git
cd ml_prediction_app```

### 2. Set Up the Virtual Environment
Create and activate a virtual environment to keep your dependencies isolated

Create: python -m venv venv
Powershell: Set-ExecutionPolicy Unrestricted -Scope Process
Activate: .\venv\Scripts\activate (on app level folder)

3. Install Dependencies
Install the required Python libraries: pip install -r requirements.txt

4. Ensure the Model is Available
Make sure you have the trained ML model (your_model.pkl) placed inside the model/ directory.

5. Run the Streamlit App
Run the app using the following command:
streamlit run start.py

The app will start, and you can open it in your browser at the URL shown in the terminal