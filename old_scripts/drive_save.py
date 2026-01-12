"""
Helper functions 
Only needed if the you use the same setup like me 
Bash usage + Google Drive logging 
"""
import os

import subprocess

def run_bash(command):
    try:
        # shell=True allows you to write the command exactly like in the terminal
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error running: {command}")

# a way to writ a bash code 
run_bash("mkdir -p ./results")
run_bash("pip install -q pydrive") 

######################################################################################
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


run_bash("echo 'Setup Complete' > setup.log")

def upload_file(filename):
    """
    Docstring для upload_file
    
    1. Copy the URL to your local browser
    2. Log in to Google
    3 .Copy the Verification Code
    4. Paste it back into terminal
    :param filename: Описание
    """
    gauth = GoogleAuth()
    
    
    gauth.CommandLineAuth() 
    
    drive = GoogleDrive(gauth)

    file_drive = drive.CreateFile({'title': filename})
    file_drive.SetContentFile(filename)
    file_drive.Upload()
    print(f"Success! File ID: {file_drive.get('id')}")

# Usage
# upload_file("hyper_last.pt")
