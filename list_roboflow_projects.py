"""
Script to list all available projects in your Roboflow workspace
"""

from roboflow import Roboflow

API_KEY = "DAWQI4w1KCHH1MlWH7t4"

def list_projects():
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    try:
        workspace = rf.workspace()
        print(f"\n📁 Workspace: {workspace.name}")
        print(f"   URL: {workspace.url}")
        print(f"\n📦 Available Projects:")
        
        # List all projects
        if hasattr(workspace, 'projects'):
            for project_path in workspace.projects:
                print(f"   - {project_path}")
        
        # Try to access the project from the screenshot
        # The URL shows: soccer-ball-kjoyy/1
        print("\n🔍 Trying to access soccer-ball-kjoyy...")
        
        # Method 1: Direct model access
        try:
            model = rf.model("soccer-ball-kjoyy/1")
            print("✅ Model accessible via rf.model('soccer-ball-kjoyy/1')")
            print(f"   Model: {model}")
        except Exception as e:
            print(f"❌ Direct model access failed: {e}")
        
        # Method 2: Try different workspace
        print("\n🔍 Checking if project is in a different workspace...")
        try:
            # Sometimes projects are in a specific workspace
            # Let's try the project name as workspace
            ws2 = rf.workspace("soccer-ball-kjoyy")
            print(f"✅ Found workspace: {ws2}")
        except Exception as e:
            print(f"❌ No workspace named 'soccer-ball-kjoyy': {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_projects()
