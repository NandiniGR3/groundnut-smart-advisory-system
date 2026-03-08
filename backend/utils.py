import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from config.settings import DRIVE_FOLDERS, DRIVE_BASE_FOLDER, LOCAL_BASE


#  Connect to Google Drive
def connect_drive():
    gauth = GoogleAuth()
    gauth.settings['client_config_file'] = "client_secret_12402204333-7p9gu7qtlvttpu2m07lol6vi42tn9uc4.apps.googleusercontent.com.json"
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive


#  Make sure local folders exist
def ensure_local_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


#  Create / get folder on Google Drive
def get_or_create_drive_folder(drive, folder_name, parent_id=None):
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        return file_list[0]['id']

    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        folder_metadata['parents'] = [{'id': parent_id}]

    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder['id']


#  Save file locally
def save_local(file_data, filename, subdir):
    folder_path = ensure_local_dir(os.path.join(LOCAL_BASE, subdir))
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        f.write(file_data)
    return file_path


#  Save any file to Drive
def save_to_drive(drive, local_path, drive_folder_id):
    file_name = os.path.basename(local_path)
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': drive_folder_id}]
    })
    file_drive.SetContentFile(local_path)
    file_drive.Upload()
    return file_drive['id']


#  Combined save (local + Drive) for Streamlit uploads or model outputs
def save_file(file, subfolder_name):
    """
    file: Streamlit uploaded file OR generated file (bytes)
    subfolder_name: 'leaf_images', 'soil_data', 'reports', etc.
    """

    # Step 1: Save locally
    filename = file.name if hasattr(file, "name") else "generated_file"
    local_path = save_local(file.read(), filename, subfolder_name)

    # Step 2: Save to Drive
    drive = connect_drive()

    # Create base folder in Drive
    base_folder_id = get_or_create_drive_folder(drive, DRIVE_BASE_FOLDER)

    # Create specific folder (e.g., leaf_images)
    drive_subfolder_id = get_or_create_drive_folder(
        drive, DRIVE_FOLDERS.get(subfolder_name, "Others"), parent_id=base_folder_id
    )

    drive_file_id = save_to_drive(drive, local_path, drive_subfolder_id)

    return {
        "local_path": local_path,
        "drive_file_id": drive_file_id
    }
# ====================================================
#  ADD THESE AT THE BOTTOM OF backend/utils.py
# ====================================================

def create_drive_structure(drive):
    
    print(" Creating or verifying Drive folder structure...")
    from config.settings import DRIVE_FOLDERS, DRIVE_BASE_FOLDER

    # Create or get the base folder
    base_folder_id = get_or_create_drive_folder(drive, DRIVE_BASE_FOLDER)
    folder_ids = {"base": base_folder_id}

    # Create or get subfolders
    for key, sub_name in DRIVE_FOLDERS.items():
        subfolder_id = get_or_create_drive_folder(drive, sub_name, parent_id=base_folder_id)
        folder_ids[key] = subfolder_id

    print(" Drive folder structure ready!")
    return folder_ids


def upload_to_drive(drive, local_path, folder_id):
    """
    Upload a file to a specific folder on Google Drive.
    Returns the uploaded file's ID.
    """
    if not os.path.exists(local_path):
        print(f" File not found: {local_path}")
        return None

    file_name = os.path.basename(local_path)
    gfile = drive.CreateFile({
        "title": file_name,
        "parents": [{"id": folder_id}]
    })
    gfile.SetContentFile(local_path)
    gfile.Upload()
    print(f" Uploaded {file_name} → Drive folder ID: {folder_id}")
    return gfile["id"]
