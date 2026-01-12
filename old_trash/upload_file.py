import sys
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def upload_to_drive(file_path):
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    
    if gauth.credentials is None:
        print("Error: no authetication")
        sys.exit(1)
        
    drive = GoogleDrive(gauth)

    filename = os.path.basename(file_path)
    print(f"Uploading {filename} to Google Drive...")
    
    file_drive = drive.CreateFile({'title': filename})
    file_drive.SetContentFile(file_path)
    file_drive.Upload()
    
    print(f"success ID:{file_drive.get('id')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_file.py <path_to_file>")
    else:
        upload_to_drive(sys.argv[1])