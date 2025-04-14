import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from data_manager import DataManager
from datetime import datetime
import pandas as pd
from offline_manager import OfflineManager
from auth_manager import AuthManager, init_session_state, login_page, signup_page

# Set environment variables to handle warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide"
)

# Initialize managers
data_manager = DataManager()
offline_manager = OfflineManager()
auth_manager = AuthManager()

# Check for offline predictions and load if available
offline_predictions = offline_manager.get_offline_predictions()
if offline_predictions:
    st.success("Loaded offline predictions.")


# Initialize session state
init_session_state()

# Care instructions database
CARE_GUIDE = {
    "watering": {
        "title": "💧 Watering Guide",
        "content": """
        - Water regularly (2-3 times per week)
        - Keep soil moist but not waterlogged
        - Water at the base of the plant
        - Best time to water: early morning
        - Reduce watering in rainy seasons
        - Check soil moisture before watering
        """
    },
    "soil": {
        "title": "🌱 Soil Requirements",
        "content": """
        - Well-draining, rich soil
        - pH between 5.0 and 6.0
        - Add organic matter regularly
        - Loamy soil is ideal
        - Avoid heavy clay soils
        - Regular soil testing recommended
        """
    },
    "sunlight": {
        "title": "☀️ Sunlight Needs",
        "content": """
        - Full sun (6-8 hours daily)
        - Partial shade in hot climates
        - Morning sun is preferable
        - Protect from intense afternoon sun
        - Ensure good air circulation
        - Use shade cloth if needed
        """
    },
    "fertilizing": {
        "title": "🌿 Fertilizing Guide",
        "content": """
        - Use balanced fertilizer (10-10-10)
        - Apply every 4-6 weeks
        - Stop fertilizing 2 weeks before harvest
        - Avoid over-fertilizing
        - Consider organic alternatives
        - Follow package instructions
        """
    },
    "diseases": {
        "title": "🔍 Disease Management",
        "content": """
        Early Blight:
        - Remove infected leaves
        - Improve air circulation
        - Use fungicides as recommended
        - Maintain plant spacing
        
        Late Blight:
        - Remove infected plants
        - Apply copper-based sprays
        - Prevent water splashing
        - Monitor weather conditions
        """
    },
    "pests": {
        "title": "🐛 Pest Control",
        "content": """
        - Regular inspection
        - Remove affected leaves
        - Use organic pesticides
        - Encourage beneficial insects
        - Maintain garden hygiene
        - Use companion planting
        """
    },
    "harvesting": {
        "title": "🥔 Harvesting Tips",
        "content": """
        - Harvest when plants yellow
        - Wait for dry weather
        - Dig carefully to avoid damage
        - Cure potatoes after harvest
        - Store in cool, dark place
        - Check for damage before storage
        """
    }
}

# Function to load and preprocess the model
@st.cache_resource
def load_model():
    try:
        # Suppress TF warnings during model loading
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model('potato_plant_model.h5')
        # Compile the model with metrics to avoid warning
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess image
def preprocess_image(image):
    try:
        # Resize image to match model input size
        image = cv2.resize(image, (256, 256))
        # Normalize pixel values
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Function to get prediction with cross-entropy
def get_prediction(model, image):
    try:
        prediction = model.predict(image, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        return class_idx, confidence, prediction[0]
    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")
        return None, None, None

# Function to display care instructions and save predictions offline
def display_care_instructions(result, image_path, prediction, confidence_scores):

    st.subheader("🌱 Care Instructions")
    
    if result == "Healthy":
        st.success("Your potato plant is healthy! Here's how to maintain it:")
        st.markdown("""
        **Watering:**
        - Water regularly (2-3 times per week)
        - Keep soil moist but not waterlogged
        - Water at the base of the plant
        
        **Soil:**
        - Well-draining, rich soil
        - pH between 5.0 and 6.0
        - Add organic matter regularly
        
        **Sunlight:**
        - Full sun (6-8 hours daily)
        - Partial shade in hot climates
        
        **Fertilizing:**
        - Use balanced fertilizer (10-10-10)
        - Apply every 4-6 weeks
        - Stop fertilizing 2 weeks before harvest
        """)
    elif result in ["Early Blight", "Late Blight"]:
        st.warning("Treatment Recommendations:")
        st.markdown("""
        **Immediate Actions:**
        - Remove infected leaves
        - Improve air circulation
        - Avoid overhead watering
        
        **Preventive Measures:**
        - Plant disease-resistant varieties
        - Rotate crops every 3-4 years
        - Maintain proper spacing
        
        **Treatment Options:**
        - Use fungicides as recommended
        - Apply copper-based sprays
        - Consider organic alternatives
        """)

def is_leaf_image(image):
    """
    Check if the image likely contains a potato leaf by analyzing:
    1. Color distribution (green shades)
    2. Edge patterns (leaf-like structure)
    3. Shape analysis (leaf-like contours)
    4. Texture analysis (leaf-like patterns)
    """
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Color Analysis
        # Define green color range for potato leaves (more specific to potato leaf green)
        lower_green = np.array([35, 50, 50])  # Adjusted for potato leaf green
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        # 2. Edge Analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Shape Analysis
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
        else:
            circularity = 0
        
        # 4. Texture Analysis
        # Calculate standard deviation of gray values (texture measure)
        texture_std = np.std(gray)
        
        # 5. Object Detection
        # Check for common non-leaf objects
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Decision making based on multiple criteria
        is_valid = True
        reason = ""
        
        # Reject if faces detected
        if len(faces) > 0:
            is_valid = False
            reason = "Human face detected in image. Please take a picture of a potato leaf only."
        
        # Check green ratio (should be significant but not too high)
        elif green_ratio < 0.15 or green_ratio > 0.85:
            is_valid = False
            reason = "Image doesn't contain enough green color typical of potato leaves."
        
        # Check edge ratio (should be moderate for leaf structure)
        elif edge_ratio < 0.01 or edge_ratio > 0.3:
            is_valid = False
            reason = "Image structure doesn't match typical leaf patterns."
        
        # Check circularity (leaves should have moderate circularity)
        elif circularity < 0.1 or circularity > 0.8:
            is_valid = False
            reason = "Image shape doesn't match typical leaf structure."
        
        # Check texture (leaves should have moderate texture variation)
        elif texture_std < 20 or texture_std > 100:
            is_valid = False
            reason = "Image texture doesn't match typical leaf patterns."
        
        return is_valid, reason
            
    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"

def search_care_instructions(query):
    """Search through care instructions and return matching results"""
    results = []
    query = query.lower()
    
    for category, info in CARE_GUIDE.items():
        if query in category.lower() or query in info["content"].lower() or query in info["title"].lower():
            results.append(info)
    
    return results

def enhance_image(image):
    """Enhance image quality before processing"""
    try:
        # Convert to float32
        image_float = np.float32(image)
        
        # Normalize
        image_normalized = cv2.normalize(image_float, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l,a,b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_image)
        
        return denoised
    except Exception as e:
        st.error(f"❌ Error enhancing image: {str(e)}")
        return image

def process_batch_images(images):
    """Process multiple images in batch"""
    results = []
    for img in images:
        try:
            # Enhance image
            enhanced = enhance_image(img)
            # Process image
            processed = preprocess_image(enhanced)
            if processed is not None:
                results.append(processed)
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    return results if results else None

def safe_predict(model, image):
    """Make predictions with error handling"""
    try:
        prediction = model.predict(image, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        return class_idx, confidence, prediction[0]
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None, None

def display_results(class_idx, confidence, probabilities, image_array, image_path, model):
    """Display classification results with confidence scores and precautions"""
    # Map class index to disease name
    class_names = ["Early Blight", "Late Blight", "Healthy"]
    
    if 0 <= class_idx < len(class_names):
        result = class_names[class_idx]
        
        # First check if it's a valid leaf image
        is_valid, reason = is_leaf_image(image_array)
        if not is_valid:
            st.error(f"❌ {reason}")
            st.info("💡 Tips for better results:")
            st.write("- Take a clear picture of a potato leaf")
            st.write("- Ensure good lighting")
            st.write("- Avoid including other objects in the frame")
            st.write("- Focus on the leaf area")
            return
        
        # Display overall result with appropriate icon and confidence
        st.markdown("## 🔍 Detection Results")
        if result == "Healthy":
            st.success(f"✅ Result: {result}")
            st.info(f"Confidence Score: {confidence:.2%}")
        else:
            st.warning(f"⚠️ Result: {result}")
            st.info(f"Confidence Score: {confidence:.2%}")
        
        # Display confidence scores for all classes
        st.markdown("### 📊 Detailed Analysis")
        for i, (disease, prob) in enumerate(zip(class_names, probabilities)):
            if disease == result:
                st.info(f"**{disease}:** {prob:.2%} 👈")
            else:
                st.write(f"**{disease}:** {prob:.2%}")
        
        # Display precautions based on the result
        st.markdown("### 🚨 Precautions & Recommendations")
        if result == "Early Blight":
            st.warning("""
            **Immediate Actions Required:**
            - Remove infected leaves immediately
            - Apply fungicide treatment
            - Improve air circulation around plants
            - Avoid overhead watering
            
            **Preventive Measures:**
            - Use disease-resistant potato varieties
            - Maintain proper plant spacing
            - Remove plant debris regularly
            - Rotate crops every 3-4 years
            
            **Treatment Options:**
            - Apply copper-based fungicides
            - Use organic alternatives like neem oil
            - Consider biological control methods
            """)
        elif result == "Late Blight":
            st.error("""
            **URGENT Actions Required:**
            - Remove and destroy infected plants
            - Apply systemic fungicide immediately
            - Isolate affected area
            - Monitor nearby plants closely
            
            **Preventive Measures:**
            - Plant certified disease-free seed potatoes
            - Ensure good drainage
            - Avoid excessive nitrogen fertilization
            - Use mulch to prevent soil splash
            
            **Treatment Options:**
            - Apply recommended fungicides
            - Consider biological controls
            - Implement strict sanitation measures
            """)
        else:  # Healthy
            st.success("""
            **Maintenance Recommendations:**
            - Continue regular watering schedule
            - Monitor for early signs of disease
            - Maintain proper fertilization
            - Keep garden clean and weed-free
            
            **Preventive Care:**
            - Regular inspection of plants
            - Proper spacing between plants
            - Good air circulation
            - Balanced nutrition
            """)
        
        # Save prediction to history if logged in
        if st.session_state.logged_in:
            prediction_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": result,
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob) for class_name, prob in zip(class_names, probabilities)
                }
            }
            auth_manager.save_to_history(st.session_state.username, prediction_data)
        else:
            # Save prediction offline if not logged in
            offline_manager.save_offline_prediction(image_path, result, confidence)
    else:
        st.error("❌ Invalid classification result")
        st.info("💡 Please try taking another picture with better lighting and focus")

def show_potato_disease_page():
    """Display the main potato disease detection page"""
    # Sidebar Navigation
    with st.sidebar:
        selected_feature = st.sidebar.radio(
            "",
            ["Disease Detection", "History", "Care Schedule", "Weather", "Settings"]
        )
        
        
        # Logout button at bottom of sidebar
        st.markdown("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

    # Main content area based on selected feature
    if selected_feature == "Disease Detection":
        st.title("🥔 Potato Disease Detection")
        st.write("""
        This app helps you detect diseases in potato leaves. It will:
        1. First check if the image contains a potato leaf
        2. Then classify any diseases present
        3. Provide care instructions based on the result
        """)

        # Load the model
        model = load_model()
        if model is None:
            return

        # Camera input
        st.subheader("📸 Take a Picture")
        camera_input = st.camera_input("Take a picture of the potato leaf")
        
        if camera_input is not None:
            try:
                # Load and enhance image
                image = Image.open(camera_input)
                image_array = np.array(image)
                enhanced_image = enhance_image(image_array)
                
                # Show original and enhanced images
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_array, caption="Original Image")
                with col2:
                    st.image(enhanced_image, caption="Enhanced Image")
                
                # Process image and make prediction
                processed_image = preprocess_image(enhanced_image)
                if processed_image is not None:
                    class_idx, confidence, probabilities = get_prediction(model, processed_image)
                    if class_idx is not None:
                        display_results(class_idx, confidence, probabilities, image_array, "live_capture.jpg", model)
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Tips for better results:")
                st.write("- Ensure good lighting")
                st.write("- Keep the camera steady")
                st.write("- Focus on the leaf area")

    elif selected_feature == "History":
        st.title("📊 Prediction History")
        if st.session_state.logged_in:
            history = auth_manager.get_user_history(st.session_state.username)
            if history:
                df = pd.DataFrame(history)
                st.dataframe(df)
                if st.button("Clear History"):
                    auth_manager.clear_history(st.session_state.username)
                    st.rerun()
            else:
                st.info("No prediction history yet")
        else:
            st.warning("Please login to view your prediction history")

    elif selected_feature == "Care Schedule":
        st.title("🌱 Care Schedule")
        st.write("Manage your potato plant care schedule and track important tasks")
        
        # Create tabs for different care aspects
        tab1, tab2, tab3 = st.tabs(["Watering Schedule", "Fertilizing Schedule", "Treatment Schedule"])
        
        with tab1:
            st.subheader("💧 Watering Schedule")
            
            # Add new watering schedule
            with st.expander("Add New Watering Schedule"):
                with st.form("watering_schedule_form"):
                    plant_name = st.text_input("Plant Name/ID")
                    frequency = st.selectbox(
                        "Watering Frequency",
                        ["Daily", "Every 2 days", "Every 3 days", "Weekly", "Custom"]
                    )
                    
                    if frequency == "Custom":
                        custom_days = st.number_input("Water every X days", min_value=1, max_value=30, value=3)
                    
                    preferred_time = st.time_input("Preferred Watering Time")
                    notes = st.text_area("Additional Notes")
                    
                    if st.form_submit_button("Add Schedule"):
                        schedule = {
                            "plant_name": plant_name,
                            "frequency": frequency if frequency != "Custom" else f"Every {custom_days} days",
                            "preferred_time": preferred_time.strftime("%H:%M"),
                            "notes": notes,
                            "last_watered": None,
                            "next_watering": None
                        }
                        data_manager.add_watering_reminder(plant_name, schedule)
                        st.success("Watering schedule added successfully!")
                        st.rerun()
            
            # Display current watering schedules
            st.subheader("Current Watering Schedules")
            upcoming = data_manager.get_upcoming_watering()
            
            if upcoming:
                for schedule in upcoming:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Plant:** {schedule['plant_id']}")
                            st.write(f"**Next Watering:** {schedule['next_watering']}")
                        with col2:
                            if st.button("Mark as Watered", key=f"water_{schedule['plant_id']}"):
                                data_manager.mark_as_watered(schedule['plant_id'])
                                st.success("Watering recorded!")
                                st.rerun()
            else:
                st.info("No active watering schedules. Add one above!")
        
        with tab2:
            st.subheader("🌿 Fertilizing Schedule")
            
            # Add new fertilizing schedule
            with st.expander("Add New Fertilizing Schedule"):
                with st.form("fertilizing_schedule_form"):
                    plant_name = st.text_input("Plant Name/ID")
                    fertilizer_type = st.selectbox(
                        "Fertilizer Type",
                        ["Balanced (10-10-10)", "High Nitrogen", "High Phosphorus", "High Potassium", "Organic"]
                    )
                    
                    frequency = st.selectbox(
                        "Fertilizing Frequency",
                        ["Every 2 weeks", "Monthly", "Every 6 weeks", "Custom"]
                    )
                    
                    if frequency == "Custom":
                        custom_weeks = st.number_input("Fertilize every X weeks", min_value=1, max_value=12, value=4)
                    
                    application_method = st.selectbox(
                        "Application Method",
                        ["Top dressing", "Side dressing", "Foliar spray", "Soil incorporation"]
                    )
                    
                    notes = st.text_area("Additional Notes")
                    
                    if st.form_submit_button("Add Schedule"):
                        schedule = {
                            "plant_name": plant_name,
                            "fertilizer_type": fertilizer_type,
                            "frequency": frequency if frequency != "Custom" else f"Every {custom_weeks} weeks",
                            "application_method": application_method,
                            "notes": notes,
                            "last_fertilized": None,
                            "next_fertilizing": None
                        }
                        data_manager.add_fertilizing_schedule(schedule)
                        st.success("Fertilizing schedule added successfully!")
                        st.rerun()
            
            # Display current fertilizing schedules
            st.subheader("Current Fertilizing Schedules")
            fertilizing_schedules = data_manager.get_upcoming_fertilizing()
            
            if fertilizing_schedules:
                for schedule in fertilizing_schedules:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Plant:** {schedule['plant_name']}")
                            st.write(f"**Fertilizer:** {schedule['fertilizer_type']}")
                            st.write(f"**Next Application:** {schedule['next_fertilizing']}")
                            st.write(f"**Method:** {schedule['application_method']}")
                        with col2:
                            if st.button("Mark as Applied", key=f"fertilize_{schedule['plant_name']}"):
                                data_manager.mark_as_fertilized(schedule['plant_name'])
                                st.success("Fertilizing recorded!")
                                st.rerun()
            else:
                st.info("No active fertilizing schedules. Add one above!")
        
        with tab3:
            st.subheader("💊 Treatment Schedule")
            
            # Add new treatment schedule
            with st.expander("Add New Treatment Schedule"):
                with st.form("treatment_schedule_form"):
                    plant_name = st.text_input("Plant Name/ID")
                    treatment_type = st.selectbox(
                        "Treatment Type",
                        ["Fungicide", "Insecticide", "Organic Treatment", "Preventive Spray", "Soil Treatment"]
                    )
                    
                    frequency = st.selectbox(
                        "Treatment Frequency",
                        ["Weekly", "Every 2 weeks", "Monthly", "Custom"]
                    )
                    
                    if frequency == "Custom":
                        custom_days = st.number_input("Treat every X days", min_value=1, max_value=90, value=14)
                    
                    application_method = st.selectbox(
                        "Application Method",
                        ["Spray", "Soil drench", "Dusting", "Systemic treatment"]
                    )
                    
                    notes = st.text_area("Additional Notes")
                    
                    if st.form_submit_button("Add Schedule"):
                        schedule = {
                            "plant_name": plant_name,
                            "treatment_type": treatment_type,
                            "frequency": frequency if frequency != "Custom" else f"Every {custom_days} days",
                            "application_method": application_method,
                            "notes": notes,
                            "last_treated": None,
                            "next_treatment": None
                        }
                        data_manager.add_treatment_schedule(schedule)
                        st.success("Treatment schedule added successfully!")
                        st.rerun()
            
            # Display current treatment schedules
            st.subheader("Current Treatment Schedules")
            treatment_schedules = data_manager.get_upcoming_treatments()
            
            if treatment_schedules:
                for schedule in treatment_schedules:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Plant:** {schedule['plant_name']}")
                            st.write(f"**Treatment:** {schedule['treatment_type']}")
                            st.write(f"**Next Application:** {schedule['next_treatment']}")
                            st.write(f"**Method:** {schedule['application_method']}")
                        with col2:
                            if st.button("Mark as Applied", key=f"treat_{schedule['plant_name']}"):
                                data_manager.mark_as_treated(schedule['plant_name'])
                                st.success("Treatment recorded!")
                                st.rerun()
            else:
                st.info("No active treatment schedules. Add one above!")
        
        # Display care history
        st.subheader("📊 Care History")
        care_history = data_manager.get_care_history()
        
        if care_history:
            df = pd.DataFrame(care_history)
            st.dataframe(df)
            
            # Export history
            if st.button("Export Care History"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="care_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No care history available yet")

    elif selected_feature == "Weather":
        st.title("🌤️ Weather Information")
        st.write("Enter your city to get weather-based care recommendations for your potato plants")
        
        # Location input
        with st.form("location_form"):
            city = st.text_input("Enter City Name", placeholder="e.g., London, New York, Tokyo")
            submit_location = st.form_submit_button("Get Weather Info")
            
            if submit_location:
                if not city.strip():
                    st.error("Please enter a city name")
                else:
                    # Get weather data
                    weather_data = data_manager.get_weather_forecast(city.strip())
                    
                    if 'error' in weather_data:
                        st.error(weather_data['error'])
                        if 'suggestions' in weather_data:
                            st.write("Suggestions:")
                            for suggestion in weather_data['suggestions']:
                                st.write(f"- {suggestion}")
                    else:
                        # Display current weather
                        st.subheader("Current Weather")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temperature", f"{weather_data['current']['temp']}°C")
                        with col2:
                            st.metric("Humidity", f"{weather_data['current']['humidity']}%")
                        with col3:
                            st.metric("Wind Speed", f"{weather_data['current']['wind_speed']} m/s")
                        
                        # Display weather-based precautions
                        st.subheader("🌱 Care Recommendations")
                        
                        # Temperature-based recommendations
                        temp = weather_data['current']['temp']
                        if temp > 30:
                            st.warning("⚠️ High Temperature Alert")
                            st.write("""
                            - Water plants more frequently
                            - Provide shade during peak hours
                            - Consider mulching to retain moisture
                            - Monitor for heat stress symptoms
                            """)
                        elif temp < 15:
                            st.warning("⚠️ Low Temperature Alert")
                            st.write("""
                            - Protect plants from frost
                            - Reduce watering frequency
                            - Consider using row covers
                            - Monitor for cold damage
                            """)
                        
                        # Humidity-based recommendations
                        humidity = weather_data['current']['humidity']
                        if humidity > 80:
                            st.warning("⚠️ High Humidity Alert")
                            st.write("""
                            - Increase air circulation
                            - Monitor for fungal diseases
                            - Avoid overhead watering
                            - Consider using fungicides preventively
                            """)
                        elif humidity < 40:
                            st.warning("⚠️ Low Humidity Alert")
                            st.write("""
                            - Increase watering frequency
                            - Use mulch to retain moisture
                            - Consider misting plants
                            - Monitor for drought stress
                            """)
                        
                        # Rain forecast recommendations
                        if any(day['rain'] > 0 for day in weather_data['daily']):
                            st.info("🌧️ Rain Forecast")
                            st.write("""
                            - Prepare for potential waterlogging
                            - Ensure proper drainage
                            - Consider covering plants if heavy rain expected
                            - Monitor for disease development
                            """)
                        
                        # General care tips based on weather
                        st.subheader("General Care Tips")
                        st.write("""
                        - Water plants early morning or late evening
                        - Monitor soil moisture regularly
                        - Keep garden tools clean and disinfected
                        - Remove any diseased leaves promptly
                        - Maintain proper spacing between plants
                        """)
                        
                        # Disease risk assessment
                        st.subheader("Disease Risk Assessment")
                        risk_factors = []
                        if temp > 25 and humidity > 70:
                            risk_factors.append("High risk of Late Blight")
                        if temp > 20 and humidity > 60:
                            risk_factors.append("Moderate risk of Early Blight")
                        
                        if risk_factors:
                            st.warning("⚠️ Disease Risk Alert")
                            for risk in risk_factors:
                                st.write(f"- {risk}")
                            st.write("""
                            Recommended Actions:
                            - Apply preventive fungicides
                            - Increase monitoring frequency
                            - Remove infected plants immediately
                            - Improve air circulation
                            """)

    elif selected_feature == "Settings":
        st.title("⚙️ Settings")
        
        # Create tabs for different settings
        tab1, tab2, tab3 = st.tabs(["Account Settings", "Theme Settings", "Notification Settings"])
        
        with tab1:
            st.subheader("👤 Account Settings")
            st.write("Update your account information")
            
            with st.form("account_settings"):
                # Get current user data
                current_user = auth_manager.users.get(st.session_state.username, {})
                
                # Username (read-only)
                st.text_input("Username", value=st.session_state.username, disabled=True)
                
                # Email
                new_email = st.text_input("Email", value=current_user.get("email", ""))
                
                # Password change
                st.write("### Change Password")
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                # Submit button
                if st.form_submit_button("Update Account"):
                    # Validate email
                    if "@" not in new_email or "." not in new_email:
                        st.error("Please enter a valid email address")
                    # Validate password change
                    elif new_password and new_password != confirm_password:
                        st.error("New passwords do not match")
                    elif new_password and current_password:
                        # Verify current password
                        if auth_manager._hash_password(current_password) != current_user.get("password"):
                            st.error("Current password is incorrect")
                        else:
                            # Update user data
                            auth_manager.users[st.session_state.username]["email"] = new_email
                            if new_password:
                                auth_manager.users[st.session_state.username]["password"] = auth_manager._hash_password(new_password)
                            auth_manager._save_users()
                            st.success("Account updated successfully!")
                            st.rerun()
                    else:
                        # Update only email
                        auth_manager.users[st.session_state.username]["email"] = new_email
                        auth_manager._save_users()
                        st.success("Email updated successfully!")
                        st.rerun()
        
        with tab2:
            st.subheader("🎨 Theme Settings")
            st.write("Customize the app's appearance")
            
            # Theme selection
            theme = st.selectbox(
                "Choose Theme",
                ["Light", "Dark", "System Default"],
                index=0
            )
            
            # Color scheme
            st.write("### Color Scheme")
            primary_color = st.color_picker("Primary Color", "#00FF00")
            accent_color = st.color_picker("Accent Color", "#FF0000")
            
            # Font size
            st.write("### Text Size")
            font_size = st.slider("Font Size", 12, 24, 16)
            
            # Apply theme button
            if st.button("Apply Theme"):
                st.success("Theme settings saved!")
                # Here you would implement the actual theme application
                st.rerun()
        
        with tab3:
            st.subheader("🔔 Notification Settings")
            st.write("Configure your notification preferences")
            
            # Email notifications
            st.write("### Email Notifications")
            email_notifications = st.checkbox("Enable email notifications", value=True, key="email_notifications_enable")
            
            if email_notifications:
                st.write("Notification Types:")
                st.checkbox("Disease detection alerts", value=True, key="email_disease_alerts")
                st.checkbox("Weather alerts", value=True, key="email_weather_alerts")
                st.checkbox("Care schedule reminders", value=True, key="email_care_reminders")
                st.checkbox("Account activity updates", value=True, key="email_account_updates")
            
            # Push notifications
            st.write("### Push Notifications")
            push_notifications = st.checkbox("Enable push notifications", value=True, key="push_notifications_enable")
            
            if push_notifications:
                st.write("Notification Types:")
                st.checkbox("Disease detection alerts", value=True, key="push_disease_alerts")
                st.checkbox("Weather alerts", value=True, key="push_weather_alerts")
                st.checkbox("Care schedule reminders", value=True, key="push_care_reminders")
            
            # Notification frequency
            st.write("### Notification Frequency")
            frequency = st.selectbox(
                "How often would you like to receive notifications?",
                ["Immediately", "Daily Summary", "Weekly Summary"],
                key="notification_frequency"
            )
            
            # Save notification settings
            if st.button("Save Notification Settings"):
                st.success("Notification settings saved!")
                # Here you would implement the actual notification settings save
                st.rerun()

def main():
    # Show login/signup if not logged in
    if not st.session_state.logged_in:
        st.title("Welcome to Potato Disease Detection 🥔")
        st.write("Please login or sign up to continue")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login_page(auth_manager)
        with tab2:
            signup_page(auth_manager)
        return
    
    # Show main app content if logged in
    show_potato_disease_page()

if __name__ == "__main__":
    main()
