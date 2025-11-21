"""
Script to download Roboflow model weights locally for offline use.
This downloads the model in YOLOv8 format so we can use it with Ultralytics directly.
"""

from roboflow import Roboflow
import os

API_KEY = "DAWQI4w1KCHH1MlWH7t4"
PROJECT_ID = "soccer-ball-kjoyy"
VERSION = 1

def download_local_model():
    print(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    try:
        # Get workspace and project
        workspace = rf.workspace()
        print(f"Workspace: {workspace}")
        
        project = workspace.project(PROJECT_ID)
        print(f"Project: {PROJECT_ID}")
        
        version = project.version(VERSION)
        print(f"Version: {VERSION}")
        
        # Download the dataset in YOLOv8 format
        # This will download the model weights and dataset structure
        print("\nDownloading model in YOLOv8 format...")
        dataset = version.download("yolov8", location="./roboflow_model")
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"Location: {os.path.abspath('./roboflow_model')}")
        print(f"\nLook for a .pt file in the downloaded folder.")
        print("You can use it with: python prototype.py --model roboflow_model/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: try to get the model directly
        try:
            model = rf.model(f"{PROJECT_ID}/{VERSION}")
            print("Model object obtained. Checking export options...")
            
            # Some Roboflow models can be exported directly
            # This might give us a local weights file
            
        except Exception as e2:
            print(f"Alternative also failed: {e2}")

if __name__ == "__main__":
    download_local_model()
