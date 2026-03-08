from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def connect_drive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("C:\Somalingaiah\minor_proj_MSc\groundnut_proj\client_secret_12402204333-7p9gu7qtlvttpu2m07lol6vi42tn9uc4.apps.googleusercontent.com.json")

    gauth.LocalWebserverAuth()

    drive = GoogleDrive(gauth)
    return drive

if __name__ == "__main__":
    drive = connect_drive()
    print("Google Drive connected successfully!")
