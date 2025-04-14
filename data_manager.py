import json
import os
from datetime import datetime
import pandas as pd
from PIL import Image
import io
import base64
import requests
from datetime import datetime, timedelta
import numpy as np

class DataManager:
    def __init__(self):
        self.history_file = 'prediction_history.json'
        self.images_dir = 'saved_images'
        self.calendar_file = 'watering_calendar.json'
        self.weather_api_key = "d5d762114c72c97d0f13d4793bbf8dfa"  # OpenWeatherMap API key
        
        # Create necessary directories and files
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Initialize history file if it doesn't exist
        if not os.path.exists(self.history_file):
            self.save_history([])
        
        # Initialize calendar file if it doesn't exist
        if not os.path.exists(self.calendar_file):
            self.save_calendar([])

    def save_prediction(self, image, prediction, confidence_scores):
        """Save prediction to history and image to disk"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"prediction_{timestamp}.jpg"
            image_path = os.path.join(self.images_dir, image_filename)
            
            # Save image
            image.save(image_path)
            
            # Create prediction entry
            prediction_entry = {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence_scores': confidence_scores,
                'image_path': image_path
            }
            
            # Load existing history
            history = self.load_history()
            
            # Add new prediction
            history.append(prediction_entry)
            
            # Save updated history
            self.save_history(history)
            
            return timestamp
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return None

    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def save_history(self, history):
        with open(self.history_file, 'w') as f:
            json.dump(history, f)

    def get_recent_predictions(self, limit=5):
        history = self.load_history()
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def add_watering_reminder(self, plant_id, watering_schedule):
        calendar = self.load_calendar()
        calendar.append({
            'plant_id': plant_id,
            'schedule': watering_schedule,
            'last_watered': None
        })
        self.save_calendar(calendar)

    def load_calendar(self):
        try:
            with open(self.calendar_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def save_calendar(self, calendar):
        with open(self.calendar_file, 'w') as f:
            json.dump(calendar, f)

    def get_upcoming_watering(self):
        calendar = self.load_calendar()
        upcoming = []
        for plant in calendar:
            if plant['last_watered']:
                last_watered = datetime.strptime(plant['last_watered'], "%Y-%m-%d")
                next_watering = last_watered + timedelta(days=plant['schedule']['frequency'])
                if next_watering > datetime.now():
                    upcoming.append({
                        'plant_id': plant['plant_id'],
                        'next_watering': next_watering.strftime("%Y-%m-%d")
                    })
        return upcoming

    def get_weather_forecast(self, city):
        """Get weather forecast for a location"""
        try:
            # Validate city name
            if not city or not isinstance(city, str) or len(city.strip()) < 2:
                return {"error": "Please enter a valid city name (at least 2 characters)"}
            
            city = city.strip()
            api_key = self.weather_api_key
            
            # First get coordinates for the city
            geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={api_key}"
            geocoding_response = requests.get(geocoding_url, timeout=10)
            
            if geocoding_response.status_code != 200:
                return {
                    "error": f"Could not find city '{city}'. Please check the spelling or try a nearby larger city",
                    "status_code": geocoding_response.status_code
                }
                
            location_data = geocoding_response.json()
            if not location_data:
                return {
                    "error": f"No matching location found for '{city}'. Try using the city's official name",
                    "suggestions": self._get_city_suggestions(city)
                }
                
            # Use first result (most relevant match)
            location = location_data[0]
            lat = location['lat']
            lon = location['lon']
            
            # Get weather data using coordinates with timeout
            weather_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=metric&appid={api_key}"
            weather_response = requests.get(weather_url, timeout=10)
            
            if weather_response.status_code != 200:
                return {
                    "error": "Weather service is currently unavailable. Please try again later",
                    "status_code": weather_response.status_code
                }
                
            weather_data = weather_response.json()
            
            # Validate weather data structure
            if 'current' not in weather_data or 'daily' not in weather_data:
                return {
                    "error": "Received incomplete weather data. Please try again",
                    "data": weather_data
                }
            
            try:
                # Format the data with additional validation
                formatted_data = {
                    'location': {
                        'city': location.get('name', city),
                        'country': location.get('country', ''),
                        'lat': lat,
                        'lon': lon
                    },
                    'current': {
                        'temp': weather_data['current']['temp'],
                        'humidity': weather_data['current']['humidity'],
                        'wind_speed': weather_data['current']['wind_speed'],
                        'weather': weather_data['current']['weather'][0]['main'],
                        'icon': weather_data['current']['weather'][0]['icon']
                    },
                    'daily': []
                }
                
                # Add daily forecast with error handling
                for day in weather_data['daily'][:5]:  # Get 5-day forecast
                    formatted_data['daily'].append({
                        'date': datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d'),
                        'temp': {
                            'day': day['temp']['day'],
                            'min': day['temp']['min'],
                            'max': day['temp']['max']
                        },
                        'humidity': day['humidity'],
                        'weather': day['weather'][0]['main'],
                        'description': day['weather'][0]['description'],
                        'icon': day['weather'][0]['icon'],
                        'rain': day.get('rain', 0),
                        'pop': day.get('pop', 0)  # Probability of precipitation
                    })
                
                return formatted_data
            except KeyError as e:
                return {
                    "error": f"Unexpected weather data format: {str(e)}",
                    "data": weather_data
                }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network error: {str(e)}",
                "details": "Please check your internet connection"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "details": "Please try again later"
            }

    def _get_city_suggestions(self, city):
        """Get possible city name suggestions"""
        suggestions = []
        if len(city) > 3:
            suggestions.append(f"Try: {city.title()}")
            suggestions.append("Check for typos in the city name")
            suggestions.append("Use the city's official English name")
            suggestions.append("Try nearby larger cities")
            
            # Common city name variations for Indian cities
            indian_cities = {
                "Chennai": ["Madras"],
                "Mumbai": ["Bombay"],
                "Kolkata": ["Calcutta"],
                "Bengaluru": ["Bangalore"],
                "Pune": ["Poona"]
            }
            
            for official_name, variants in indian_cities.items():
                if city.lower() in [v.lower() for v in variants]:
                    suggestions.append(f"Try the official name: {official_name}")
                    break
                
        return suggestions

    def get_care_recommendations(self, weather_data):
        if not weather_data:
            return "Unable to get weather data for recommendations."
        
        recommendations = []
        for day in weather_data:
            if day['temperature'] > 30:
                recommendations.append(f"High temperature ({day['temperature']}°C) on {day['date']}. Consider extra watering.")
            if day['humidity'] < 40:
                recommendations.append(f"Low humidity ({day['humidity']}%) on {day['date']}. Increase watering frequency.")
            if 'rain' in day['description'].lower():
                recommendations.append(f"Rain expected on {day['date']}. Adjust watering schedule accordingly.")
        
        return recommendations if recommendations else "Weather conditions are favorable for normal watering schedule."

    def add_fertilizing_schedule(self, schedule):
        """Add a new fertilizing schedule"""
        calendar = self.load_calendar()
        calendar.append({
            'type': 'fertilizing',
            'schedule': schedule
        })
        self.save_calendar(calendar)

    def add_treatment_schedule(self, schedule):
        """Add a new treatment schedule"""
        calendar = self.load_calendar()
        calendar.append({
            'type': 'treatment',
            'schedule': schedule
        })
        self.save_calendar(calendar)

    def get_upcoming_fertilizing(self):
        """Get upcoming fertilizing schedules"""
        calendar = self.load_calendar()
        upcoming = []
        for item in calendar:
            if item['type'] == 'fertilizing':
                schedule = item['schedule']
                if schedule['last_fertilized']:
                    last_fertilized = datetime.strptime(schedule['last_fertilized'], "%Y-%m-%d")
                    next_fertilizing = last_fertilized + timedelta(days=self._parse_frequency(schedule['frequency']))
                    if next_fertilizing > datetime.now():
                        upcoming.append(schedule)
                else:
                    upcoming.append(schedule)
        return upcoming

    def get_upcoming_treatments(self):
        """Get upcoming treatment schedules"""
        calendar = self.load_calendar()
        upcoming = []
        for item in calendar:
            if item['type'] == 'treatment':
                schedule = item['schedule']
                if schedule['last_treated']:
                    last_treated = datetime.strptime(schedule['last_treated'], "%Y-%m-%d")
                    next_treatment = last_treated + timedelta(days=self._parse_frequency(schedule['frequency']))
                    if next_treatment > datetime.now():
                        upcoming.append(schedule)
                else:
                    upcoming.append(schedule)
        return upcoming

    def mark_as_fertilized(self, plant_name):
        """Mark a plant as fertilized"""
        calendar = self.load_calendar()
        for item in calendar:
            if item['type'] == 'fertilizing' and item['schedule']['plant_name'] == plant_name:
                item['schedule']['last_fertilized'] = datetime.now().strftime("%Y-%m-%d")
                self.save_calendar(calendar)
                break

    def mark_as_treated(self, plant_name):
        """Mark a plant as treated"""
        calendar = self.load_calendar()
        for item in calendar:
            if item['type'] == 'treatment' and item['schedule']['plant_name'] == plant_name:
                item['schedule']['last_treated'] = datetime.now().strftime("%Y-%m-%d")
                self.save_calendar(calendar)
                break

    def get_care_history(self):
        """Get complete care history"""
        calendar = self.load_calendar()
        history = []
        for item in calendar:
            if item['type'] == 'watering':
                history.append({
                    'date': item['schedule'].get('last_watered'),
                    'type': 'Watering',
                    'plant': item['schedule']['plant_name'],
                    'details': f"Frequency: {item['schedule']['frequency']}"
                })
            elif item['type'] == 'fertilizing':
                history.append({
                    'date': item['schedule'].get('last_fertilized'),
                    'type': 'Fertilizing',
                    'plant': item['schedule']['plant_name'],
                    'details': f"Type: {item['schedule']['fertilizer_type']}, Method: {item['schedule']['application_method']}"
                })
            elif item['type'] == 'treatment':
                history.append({
                    'date': item['schedule'].get('last_treated'),
                    'type': 'Treatment',
                    'plant': item['schedule']['plant_name'],
                    'details': f"Type: {item['schedule']['treatment_type']}, Method: {item['schedule']['application_method']}"
                })
        return sorted(history, key=lambda x: x['date'] if x['date'] else '', reverse=True)

    def _parse_frequency(self, frequency):
        """Convert frequency string to number of days"""
        if frequency == "Daily":
            return 1
        elif frequency == "Every 2 days":
            return 2
        elif frequency == "Every 3 days":
            return 3
        elif frequency == "Weekly":
            return 7
        elif frequency == "Every 2 weeks":
            return 14
        elif frequency == "Monthly":
            return 30
        elif frequency == "Every 6 weeks":
            return 42
        elif frequency.startswith("Every "):
            try:
                return int(frequency.split()[1])
            except:
                return 7  # Default to weekly
        return 7  # Default to weekly 