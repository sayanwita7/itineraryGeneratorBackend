from flask import Flask,request,redirect,session,url_for,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from pymongo import MongoClient
import os
import certifi
from dotenv import load_dotenv
import re
from model.kolkata_itinerary import fetch_itinerary

load_dotenv()
uri=os.getenv("MONGO_DB_URL")
client=MongoClient(uri,tlsCAFile=certifi.where())

DATABASE_NAME="ItineraryGenerator"

db = client[DATABASE_NAME]
users_col = db["User"]
itinerary_col = db["Itinerary"]
saved_col = db["UserItinerary"]
hotel_col=db["Hotels"]
app=Flask(__name__)
app.secret_key="Hello"
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "Welcome"})


@app.route("/user/register", methods=['POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()

        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        uname = data.get('username', '').strip()
        phno = data.get('phone', '').strip()
        password = data.get('password', '')

        if not all([name, email, uname, phno, password]):
            return jsonify({"error": "All fields are required."}), 400

        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_pattern, email):
            return jsonify({"error": "Invalid email format."}), 400

        if not phno.isdigit() or len(phno) < 10 or len(phno) > 15:
            return jsonify({"error": "Invalid phone number."}), 400

        password_pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@#$!%*?&]).{8,}$"
        if not re.match(password_pattern, password):
            return jsonify({
                "error": "Password must be at least 8 characters long, include uppercase, lowercase, number, and special character."
            }), 400

        if users_col.find_one({"username": uname}):
            return jsonify({"error": "Account already exists... Please Log In!"}), 409

        # Insert user
        users_col.insert_one({
            "name": name,
            "email": email,
            "username": uname,
            "phone": int(phno),
            "password": generate_password_hash(password)
        })

        return jsonify({"message": "Account successfully created!!"}), 201


@app.route("/user/login", methods=['POST'])
def logIn():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = users_col.find_one({"username": username})
        if user:
            if check_password_hash(user['password'], password):
                session['user'] = user['username']
                user_id = str(user['_id'])
                return jsonify({
                    "message": "Log In successful!",
                    "userId": user_id,
                    "username": user['username']
                }),200
            else:
                return jsonify({"Error": "Incorrect Email or Password..."})
        else:
            return jsonify({"Error": "Email does not exist... Please Sign Up first!"})


@app.route("/user/logout")
def logOut():
    session.pop('user', None)
    return jsonify({"message": "Logged out successfully"}),200

# ---------------------- PREDICT ITINERARY ----------------------
@app.route("/itinerary/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        duration = data.get('duration')
        budget = data.get('budget')
        grouptype = data.get('grouptype')
        arrival = data.get('arrival')
        start_location="Hotel"
        
        suggestion = itinerary_col.find_one({
            "Duration": duration,
            "Budget": budget,
            "GroupType": grouptype,
            "Arrival":arrival
        })
        if suggestion:
            return jsonify({
                "message": "Itinerary Generated",
                "Hotel": suggestion["Suggestions"]["Hotel"],   # nested
                "Itinerary": suggestion["Suggestions"]["Itinerary"]  # nested
            })
        else:
            res=fetch_itinerary(duration, budget, grouptype, arrival, start_location)
            itinerary_col.insert_one({
                "Duration": duration,
                "Budget": budget,
                "GroupType": grouptype,
                "Arrival":arrival,
                "Suggestions":res
            })
            return jsonify({"message":"Itinerary Generated", "Hotel":res["Hotel"], "Itinerary":res["Itinerary"] })

# ---------------------- SAVE ITINERARY ----------------------
@app.route("/itinerary/save", methods=['POST'])
def saveItinerary():
    if request.method == 'POST':
        data = request.get_json()
        duration = data.get('duration')
        budget = data.get('budget')
        grouptype = data.get('grouptype')

        itinerary_doc = itinerary_col.find_one({
            "Duration": duration,
            "Budget": budget,
            "GroupType": grouptype
        })

        user_doc = users_col.find_one({"Username": session.get('user')})

        if itinerary_doc and user_doc:
            saved_col.insert_one({
                "ItineraryId": itinerary_doc['_id'],
                "UserId": user_doc['_id']
            })
            return jsonify({'message': "Data saved"})
        else:
            return jsonify({"Error": "Not Found"}), 404

if __name__ == '__main__':
    app.run(debug=True)