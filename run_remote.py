import subprocess
import time
import os
import requests
import json

def get_ngrok_url():
    """Get the ngrok URL from the API"""
    retries = 10
    while retries > 0:
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()['tunnels']
            if tunnels:
                return tunnels[0]['public_url']
        except:
            retries -= 1
            time.sleep(1)
    return None

def main():
    print("\n🚀 Starting Potato Disease Detection App...")
    
    try:
        # Start Streamlit
        print("\n📱 Starting Streamlit server...")
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for Streamlit to start and get its URLs
        time.sleep(3)
        print("\n✅ Streamlit is running!")
        print("💻 Local URL: http://localhost:8501")
        
        # Start ngrok
        print("\n🔄 Setting up remote access...")
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start and get the URL
        time.sleep(3)
        ngrok_url = get_ngrok_url()
        
        if ngrok_url:
            print("\n✨ Remote access setup complete!")
            print("\n📱 Access URLs:")
            print(f"Local URL:     http://localhost:8501")
            print(f"Public URL:    {ngrok_url}")
            print("\n🌟 You can use any of these URLs to access your app")
            print("💡 Use the Public URL to access from your phone or other devices")
            print("\n⚠️ Keep this window open to maintain the connection")
            print("❌ Press Ctrl+C to stop the app")
            
            # Keep the app running
            while True:
                time.sleep(1)
                
        else:
            print("\n❌ Could not get public URL")
            print("💡 You can still use: http://localhost:8501")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping the app...")
        try:
            streamlit_process.terminate()
            ngrok_process.terminate()
            os.system('taskkill /f /im ngrok.exe 2>nul')
        except:
            pass
        print("✅ App stopped successfully!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure you have an internet connection")
        print("2. Check if port 8501 is available")
        print("3. Try restarting the app")

if __name__ == "__main__":
    main() 