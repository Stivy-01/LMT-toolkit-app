import os
import subprocess
import sys

def run_streamlit_app():
    """
    Run the Streamlit app from the current directory
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, "streamlit_app", "app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Streamlit app file not found at {app_path}")
        return
    
    print("Starting LMT Toolkit App...")
    print(f"App location: {app_path}")
    print("Data directory is configured in streamlit_app/config.py")
    print("Ctrl+C to stop the app")
    
    # Run the Streamlit app
    try:
        subprocess.run([
            "streamlit", "run", 
            app_path, 
            "--browser.serverAddress=localhost",
            "--server.port=8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\nStopping app...")
    except Exception as e:
        print(f"Error starting Streamlit app: {e}")
        
        # Check if streamlit is installed
        try:
            subprocess.run(["streamlit", "--version"], check=True, capture_output=True)
        except:
            print("\nStreamlit does not appear to be installed.")
            print("Please install it with: pip install streamlit")

if __name__ == "__main__":
    run_streamlit_app() 