import json
import os
import hashlib
import streamlit as st
from datetime import datetime

class AuthManager:
    def __init__(self):
        self.users_file = "data/users.json"
        self._ensure_users_file()
        self.users = self._load_users()

    def _ensure_users_file(self):
        """Create users file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)

    def _load_users(self):
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password, email):
        """Register a new user"""
        if username in self.users:
            return False, "Username already exists"
        
        # Hash the password before storing
        hashed_password = self._hash_password(password)
        
        self.users[username] = {
            "password": hashed_password,
            "email": email,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": []
        }
        self._save_users()
        return True, "Account created successfully"

    def login(self, username, password):
        """Login a user"""
        if username not in self.users:
            return False, "Username not found! Please sign up first"
        
        # Hash the input password and compare with stored hash
        input_password_hash = self._hash_password(password)
        stored_password_hash = self.users[username]["password"]
        
        if input_password_hash != stored_password_hash:
            return False, "Incorrect password! Please try again"
        
        return True, "Login successful"

    def get_user_history(self, username):
        """Get user's prediction history"""
        if username in self.users:
            return self.users[username].get("history", [])
        return []

    def save_to_history(self, username, prediction_data):
        """Save prediction to user's history"""
        if username in self.users:
            self.users[username]["history"].append(prediction_data)
            self._save_users()
            return True
        return False

    def clear_history(self, username):
        """Clear user's prediction history"""
        if username in self.users:
            self.users[username]["history"] = []
            self._save_users()
            return True
        return False

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

def login_page(auth_manager):
    """Display login page"""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, message = auth_manager.login(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(message)
                st.rerun()
            else:
                st.error(message)

def signup_page(auth_manager):
    """Display signup page"""
    st.title("Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = auth_manager.signup(username, password, email)
                if success:
                    st.success(message)
                    st.info("Please login with your new account")
                    st.rerun()
                else:
                    st.error(message) 