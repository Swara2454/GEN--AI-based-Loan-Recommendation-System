from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import numpy as np
import pandas as pd
import uuid
import re
from difflib import get_close_matches
import google.generativeai as genai
import config  # contains GEMINI_API_KEY

# -------------------- Flask App --------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app)

# -------------------- Configure Gemini --------------------
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------- Load Data & Models --------------------
dataset = pd.read_csv("loan_reco_merged_dataset_updated_final.csv")
model = joblib.load('loan_type_model.pkl')
le_target = joblib.load('label_encoder_target.pkl')
label_encoders = joblib.load('label_encoders.pkl')

feature_cols = [
    'Age', 'MonthlyIncome', 'PreferredTenure_Yeras', 'Purpose', 'EmploymentStatus',
    'Education', 'MaritalStatus', 'Gender', 'PreviousLoans', 'ExistingEMIs',
    'Expenses', 'RepaymentCapacity', 'LoanAmount'
]

# -------------------- Database --------------------
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize users table
conn = get_db_connection()
conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password TEXT NOT NULL,
    mobile_no TEXT
)
''')
conn.commit()
conn.close()

# -------------------- Loan Q&A --------------------
user_sessions = {}

questions = [
    ('Age', "How old are you?"),
    ('MonthlyIncome', "What's your monthly income?"),
    ('PreferredTenure_Yeras', "How many years do you want the loan for?"),
    ('Purpose', "What's the purpose of the loan? (Medical, Travel, Business, Car, etc.)"),
    ('EmploymentStatus', "What's your employment status? (Salaried, Self-Employed, Student, etc.)"),
    ('Education', "What's your highest education level?"),
    ('MaritalStatus', "Can you tell me your marital status?"),
    ('Gender', "What gender do you identify as?"),
    ('PreviousLoans', "How many loans have you taken before?"),
    ('ExistingEMIs', "What’s your current total EMI amount?"),
    ('Expenses', "What are your monthly expenses?"),
    ('RepaymentCapacity', "Roughly how much can you comfortably repay per month?"),
    ('LoanAmount', "Finally, how much loan are you seeking?")
]

purpose_map = {
    # Auto Loans
    "car": "Auto Loan", "vehicle": "Auto Loan", "bike": "Auto Loan", "auto": "Auto Loan",
    
    # Education
    "education": "Education Loan", "study": "Education Loan", "school": "Education Loan", 
    "college": "Education Loan", "university": "Education Loan",
    
    # Medical
    "medical": "Medical Loan", "hospital": "Medical Loan", "health": "Medical Loan", 
    "treatment": "Medical Loan",
    
    # Travel
    "travel": "Travel Loan", "vacation": "Travel Loan", "tour": "Travel Loan", "trip": "Travel Loan",
    
    # Business
    "business": "Business Loan", "startup": "Business Loan", "shop": "Business Loan", "office": "Business Loan",
    
    # Agriculture
    "agriculture": "Agri Loan", "farming": "Agri Loan", "crop": "Agri Loan", "farmer": "Agri Loan",
    
    # Home
    "home": "Home Loan", "house": "Home Loan", "property": "Home Loan", "flat": "Home Loan",
    
    # Personal
    "personal": "Personal Loan", "cash": "Personal Loan", "urgent": "Personal Loan", "emergency": "Personal Loan",
    
    # Wedding
    "wedding": "Wedding Loan", "marriage": "Wedding Loan",
    
    # Debt consolidation
    "debt": "DebtConsolidationLoan", "consolidation": "DebtConsolidationLoan",
    
    # Home improvement
    "renovation": "Home Improvement Loan", "repair": "Home Improvement Loan", 
    "improvement": "Home Improvement Loan",
    
    # Consumer durable
    "electronics": "Consumer Durable Loan", "gadgets": "Consumer Durable Loan", 
    "appliances": "Consumer Durable Loan"
}

# -------------------- Helper Functions --------------------
def fuzzy_match(value, valid_options, cutoff=0.6):
    if not value or not isinstance(value, str):
        return value
    value = value.lower().strip()
    valid_options = [str(v).lower() for v in valid_options]
    match = get_close_matches(value, valid_options, n=1, cutoff=cutoff)
    return match[0] if match else value

def normalize_purpose(user_input):
    words = re.findall(r'\w+', user_input.lower())
    for w in words:
        if w in purpose_map:
            return purpose_map[w]
    valid_purposes = dataset["LoanType"].dropna().unique()
    return fuzzy_match(user_input, valid_purposes)

def predict_loan_type(profile):
    if "Purpose" in profile and profile["Purpose"]:
        return profile["Purpose"]
    features = []
    for col in feature_cols:
        val = profile.get(col)
        if val is None:
            raise ValueError(f"Missing feature: {col}")
        if col in label_encoders:
            le = label_encoders[col]
            val = fuzzy_match(str(val), le.classes_)
            if val not in le.classes_:
                val = le.classes_[0]
            val = le.transform([val])[0]
        else:
            val = float(val)
        features.append(val)
    X = np.array(features).reshape(1, -1)
    pred_encoded = model.predict(X)[0]
    return le_target.inverse_transform([pred_encoded])[0]

def generate_recommendation(profile, loan_type):
    loan_type = normalize_purpose(loan_type)

    # Filter dataset for matching loan type
    loan_data = dataset[dataset["LoanType"].str.lower().str.contains(loan_type.lower(), na=False)]
    if loan_data.empty:
        return {"error": True, "message": f"❌ No loan schemes available for {loan_type}."}

    # Extract loan constraints
    min_amount = int(loan_data["LoanAmount"].min())
    max_amount = int(loan_data["LoanAmount"].max())
    avg_tenure = int(loan_data["PreferredTenure_Yeras"].mean())
    interest_rate = round(loan_data["InterestRate"].mean(), 2) if "InterestRate" in loan_data else None

    # User inputs
    loan_amount = float(profile['LoanAmount'])
    repayment_capacity = float(profile['RepaymentCapacity'])

    # Amount eligibility checks
    if loan_amount < min_amount:
        return {"error": True, "message": f"❌ Minimum loan for {loan_type} is {min_amount}. Your request ({loan_amount}) is too low."}
    if loan_amount > max_amount:
        return {"error": True, "message": f"❌ Maximum loan for {loan_type} is {max_amount}. Your request ({loan_amount}) is too high."}

    # Recommendation logic
    recommended_amount = min(loan_amount, max_amount)
    recommended_tenure = max(1, avg_tenure)
    monthly_installment = recommended_amount / (recommended_tenure * 12)

    # Adjust tenure if EMI > repayment capacity
    if monthly_installment > repayment_capacity:
        recommended_tenure = int((recommended_amount / repayment_capacity) / 12) + 1
        monthly_installment = recommended_amount / (recommended_tenure * 12)

    # Final recommendation
    result = {
        "error": False,
        "loan_type": loan_type,
        "recommended_amount": int(recommended_amount),
        "amount_range": f"{min_amount}-{max_amount}",
        "recommended_tenure": recommended_tenure,
        "monthly_installment": int(monthly_installment)
    }

    if interest_rate:
        result["interest_rate"] = f"{interest_rate} %"

    return result

def gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# -------------------- Auth Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email_id']
        name = request.form['name']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        mobile_no = request.form['mobile_no']
        if password != confirm_password:
            return "Passwords do not match!"

        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (email, name, password, mobile_no) VALUES (?, ?, ?, ?)',
                (email, name, password, mobile_no)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('login'))  
        except sqlite3.IntegrityError:
            conn.close()
            return "Email already exists!"

    return render_template('signup.html')   

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email_id']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE email = ? AND password = ?',
            (email, password)
        ).fetchone()
        conn.close()

        if user:
            session['user'] = user['name']
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "Invalid email or password!"})

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

# -------------------- Chatbot Routes --------------------
@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    session_id = session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['chat_session_id'] = session_id

    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'profile': {},
            'current_question': 0,
            'completed': False,
            'chat_mode': False
        }

    session_data = user_sessions[session_id]
    msg = request.json.get('message', '').strip()

    # Exit keywords
    exit_keywords = ["exit", "quit", "bye", "no", "stop", "cancel"]
    if any(word in msg.lower() for word in exit_keywords):
        user_sessions.pop(session_id, None)
        return jsonify({
            'reply': "👋 Thank you for chatting with LoanBot! Have a great day! 🌟",
            'done': True
        })

    greetings = ["hi", "hello", "hey", "hey there"]

    # Initial greeting
    if session_data['current_question'] == 0 and not session_data['completed']:
        if any(g in msg.lower() for g in greetings):
            reply = "Hi there! How are you doing today? 😊 Would you like some help with loans?"
            return jsonify({'reply': reply, 'done': False})
        elif "yes" in msg.lower():
            session_data['current_question'] = 1
            key, question_text = questions[0]
            prompt = f"Friendly: Ask the user this question naturally: {question_text}"
            reply = gemini_response(prompt)
            return jsonify({'reply': reply, 'done': False})

    # Structured Q&A
    if 1 <= session_data['current_question'] <= len(questions):
        key, question_text = questions[session_data['current_question'] - 1]

        # Numeric fields
        if key in ['Age', 'MonthlyIncome', 'PreferredTenure_Yeras', 'PreviousLoans',
                   'ExistingEMIs', 'Expenses', 'RepaymentCapacity', 'LoanAmount']:
            numbers = re.findall(r'\d+\.?\d*', msg)
            if numbers:
                session_data['profile'][key] = float(numbers[0])

                # ---- Eligibility Checks (Friendly Restart) ----
                if key == "Age" and session_data['profile'][key] < 18:
                    user_sessions.pop(session_id, None)
                    session['chat_session_id'] = str(uuid.uuid4())
                    return jsonify({
                        'reply': "❌ Sorry, you must be at least 18 years old to apply for a loan.\n\n🔄 Let's start fresh! Hi there 😊 Would you like some help with loans?",
                        'done': False
                    })

                if key == "MonthlyIncome" and session_data['profile'][key] <= 0:
                    user_sessions.pop(session_id, None)
                    session['chat_session_id'] = str(uuid.uuid4())
                    return jsonify({
                        'reply': "❌ Sorry, you must have a valid monthly income to be eligible for a loan.\n\n🔄 Let's start fresh! Hi there 😊 Would you like some help with loans?",
                        'done': False
                    })

            else:
                return jsonify({'reply': f"Oops! I need a number for {key}. Could you provide that?", 'done': False})
        else:
            if key in label_encoders:
                valid_options = label_encoders[key].classes_
                session_data['profile'][key] = fuzzy_match(msg, valid_options)
            else:
                session_data['profile'][key] = msg

        session_data['current_question'] += 1

        if session_data['current_question'] <= len(questions):
            next_key, next_q = questions[session_data['current_question'] - 1]
            prompt = f"Friendly: Ask the user this question naturally: {next_q}"
            reply = gemini_response(prompt)
            return jsonify({'reply': reply, 'done': False})
        else:
            try:
                loan_type = predict_loan_type(session_data['profile'])
                recommendation = generate_recommendation(session_data['profile'], loan_type)
                if recommendation.get("error"):
                    reply_text = recommendation["message"]
                else:
                    prompt = f"Friendly: Present this loan recommendation naturally: {recommendation}"
                    reply_text = gemini_response(prompt)

                session_data['completed'] = True
                session_data['chat_mode'] = True
                return jsonify({'reply': reply_text, 'done': True})
            except Exception as e:
                return jsonify({'reply': f"⚠️ Something went wrong: {str(e)}", 'done': True})

    # Free chat
    if session_data.get('chat_mode', False):
        prompt = f"You are a friendly AI chatbot. Continue the conversation naturally with the user. User said: {msg}"
        reply_text = gemini_response(prompt)
        return jsonify({'reply': reply_text, 'done': False})

    return jsonify({'reply': "Oops! I didn't understand that. Type 'loan recommendation' to start.", 'done': False})

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)
