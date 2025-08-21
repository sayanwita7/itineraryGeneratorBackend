Tripzy - AI-Powered Itinerary Generator
Tripzy is a personalized trip planning tool built using Streamlit (Frontend) and Flask (Backend).  
It generates customized travel itineraries for touring around Kolkata based on preferences like trip duration, budget, travel group and arrival location.

Project Structure for backend using Flask:
1. app.py: Main backend server
2. model : ML model
3. requirements.txt : Backend dependencies
4. .env : Backend environment variables

Project Setup Guide:
1. Clone the Repository and navigate to the directory:
git clone https://github.com/sayanwita7/itineraryGeneratorBackend.git
cd itinerary-generator
2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
3. Install dependencies:
pip install -r requirements.txt
4. Run the Flask server:
python app.py

Environment variables:
MONGO_DB_URL
 
