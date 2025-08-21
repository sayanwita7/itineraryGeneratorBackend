from flask import Flask,request,redirect,session,url_for,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from pymongo import MongoClient
import os
import certifi
from dotenv import load_dotenv
import re
from model.kolkata_itinerary import fetch_itinerary
from bson.objectid import ObjectId
from bson import ObjectId

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
    budget_value = {"Low": 1000, "Mid": 3000, "High": 5000}
    data = request.get_json()
    duration = data.get('duration')
    budget = data.get('budget')
    grouptype = data.get('grouptype')
    arrival = data.get('arrival')
    username = session.get('user')
    start_location = "Hotel"
    suggestion = itinerary_col.find_one({
        "Duration": duration,
        "Budget": budget,
        "GroupType": grouptype,
        "Arrival": arrival
    })
    
    if suggestion:
        itinerary_id = suggestion["_id"]
        hotel_suggestion = suggestion["Suggestions"]["Hotel"]
        itinerary_suggestion = suggestion["Suggestions"]["Itinerary"]
        existing = saved_col.find_one({
            "username": username,
            "itinerary_id": itinerary_id
        })
        if not existing:
            saved_col.insert_one({
                "username": username,
                "itinerary_id": itinerary_id
            })
        x=0
    else:
        res = fetch_itinerary(duration, budget_value[budget], grouptype, arrival, start_location)
        result = itinerary_col.insert_one({
            "Duration": duration,
            "Budget": budget,
            "GroupType": grouptype,
            "Arrival": arrival,
            "Suggestions": res,
        })
        itinerary_id = result.inserted_id
        hotel_suggestion = res["Hotel"]
        itinerary_suggestion = res["Itinerary"]
        saved_col.insert_one({
            "username": username,
            "itinerary_id": itinerary_id,
        })
    
    return jsonify({
        "message": "Itinerary Generated",
        "Hotel": hotel_suggestion,
        "Itinerary": itinerary_suggestion
    })

        

# ---------------------- USER HISTORY ----------------------

@app.route("/user/history", methods=['POST'])
def userHistory():
    # username = session.get('user')
    data = request.get_json()
    username=data.get('username')
    if not username:
        return jsonify({"error": "User not logged in"}), 401
    user = users_col.find_one({"username": username}, {"password": 0})
    if not user:
        return jsonify({"error": "User not found"}), 404
    user_info = {
        "name": user.get("name"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "username": user.get("username")
    }
    user_itineraries = saved_col.find({"username": username})
    itinerary_list = []
    for ui in user_itineraries:
        itinerary_id = ui.get("itinerary_id")
        if itinerary_id:
            itinerary = itinerary_col.find_one({"_id": ObjectId(itinerary_id)})
            if itinerary:
                itinerary_list.append({
                    "itinerary_id": str(itinerary["_id"]),
                    "Duration": itinerary.get("Duration"),
                    "Budget": itinerary.get("Budget"),
                    "GroupType": itinerary.get("GroupType"),
                    "Arrival": itinerary.get("Arrival")
                })
    response = {
        "user_info": user_info,
        "itineraries": itinerary_list
    }
    return jsonify(response), 200

@app.route("/user/fetch_itinerary/<itinerary_id>", methods=["GET"])
def fetchItinerary(itinerary_id):
    try:
        itinerary = itinerary_col.find_one({"_id": ObjectId(itinerary_id)})
        if not itinerary:
            return jsonify({"error": "Itinerary not found"}), 404
        itinerary["_id"] = str(itinerary["_id"])
        hotel_suggestion = itinerary.get("Suggestions", {}).get("Hotel", "")
        itinerary_suggestion = itinerary.get("Suggestions", {}).get("Itinerary", {})
        return jsonify({
            "message": "Itinerary fetched successfully",
            "Hotel": hotel_suggestion,
            "Itinerary": itinerary_suggestion
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)