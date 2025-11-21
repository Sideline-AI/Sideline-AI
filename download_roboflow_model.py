from roboflow import Roboflow
import sys

# User provided API Key
API_KEY = "DAWQI4w1KCHH1MlWH7t4"
PROJECT_ID = "soccer-ball-kjoyy"
VERSION_NUMBER = 1

def download_model():
    print(f"Authenticating with Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    # We need to find the workspace. 
    # Usually the project ID in the URL is 'workspace/project'.
    # But here we only have 'soccer-ball-kjoyy'. 
    # Let's try to list workspaces or just guess the project is in the default workspace associated with the key.
    
    # However, rf.workspace() without arguments gets the default workspace?
    # Let's try to access the project directly if possible, or iterate workspaces.
    
    print("Listing workspaces...")
    try:
        # This is a bit of a guess on how the library handles it if we don't know the workspace name
        # But usually we can just do:
        workspace = rf.workspace() # Gets the default or first one?
        print(f"Default workspace: {workspace}")
        
        project = workspace.project(PROJECT_ID)
        version = project.version(VERSION_NUMBER)
        
        print(f"Found project: {PROJECT_ID}, Version: {VERSION_NUMBER}")
        print("Downloading YOLOv8 weights...")
        
        # download("yolov8") returns the path to the dataset/weights
        # We specifically want the .pt file.
        # Sometimes download() downloads the dataset. 
        # We want the model weights.
        
        # version.deploy(model_type="yolov8", model_path=".") # This might be for deploying?
        
        # To get weights, we often use:
        version.download("yolov8", location="roboflow_model")
        print("Download complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback: try to list workspaces to help debug
        # (The user might need to provide the workspace name if it's not the default)
        sys.exit(1)

if __name__ == "__main__":
    download_model()
