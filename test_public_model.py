"""
Download public Roboflow model for soccer ball detection
"""

from roboflow import Roboflow

# For public models, we can use a different approach
API_KEY = "DAWQI4w1KCHH1MlWH7t4"

def download_public_model():
    print("Attempting to download public soccer ball model...")
    rf = Roboflow(api_key=API_KEY)
    
    # Try to access as a public model
    # Format: workspace/project/version
    try:
        print("\n🔍 Method 1: Direct universe access...")
        # The screenshot shows: soccer-ball-kjoyy/1
        # This might be in Roboflow Universe (public models)
        model = rf.model("soccer-ball-kjoyy/1")
        print(f"✅ Model found: {model}")
        
        # Try to download/export
        print("\nAttempting to get model info...")
        print(f"Model ID: soccer-ball-kjoyy/1")
        
        # For inference, we can use it directly (but it will be slow via API)
        # Let's see if we can export it
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
        # Try alternative format
        print("\n🔍 Method 2: Trying with workspace prefix...")
        try:
            # Sometimes public models need the full path
            model = rf.workspace().project("soccer-ball-kjoyy").version(1).model
            print(f"✅ Model found via workspace: {model}")
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            print("\n💡 Suggestion: This model might be private or require different credentials.")
            print("   You can:")
            print("   1. Train your own model on Roboflow with your footage")
            print("   2. Use the existing YOLOv8 models (yolov8s.pt, yolov8m.pt)")
            print("   3. Fine-tune YOLOv8 on soccer ball images")

if __name__ == "__main__":
    download_public_model()
