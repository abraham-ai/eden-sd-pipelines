import os, sys, shutil
from pathlib import Path
import requests
import zipfile
from mimetypes import guess_extension
from PIL import Image
import eden_utils

def run_and_kill_cmd(command, pipe_output=True):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.25)

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()    
    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd, stdout:")
            print(stdout)
        if stderr:
            print("cmd, stderr:")
            print(stderr)

    p.send_signal(signal.SIGTERM) # Sends termination signal
    p.wait()  # Waits for process to terminate

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()

    # If the process hasn't ended yet
    if p.poll() is None:  
        p.kill()  # Forcefully kill the process
        p.wait()  # Wait for the process to terminate

    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd done, stdout:")
            print(stdout)
        if stderr:
            print("cmd done, stderr:")
            print(stderr)

def download(url, folder, filepath = None):
    """
    Robustly download a file from a given URL to the specified folder, automatically infering the file extension.
    
    Args:
        url (str):      The URL of the file to download.
        folder (str):   The folder where the downloaded file should be saved.
        filepath (str): (Optional) The path to the downloaded file. If None, the path will be inferred from the URL.
        
    Returns:
        filepath (Path): The path to the downloaded file.

    """
    try:
        folder_path = Path(folder)
        
        if filepath is None:
            # Make a preliminary request to the URL to get the content type without downloading the file
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('Content-Type')
            
            # Guess file extension based on the content type
            ext = guess_extension(content_type) or ''  # Default to empty string if extension not found
            # Parse the URL to get the filename and append the extension (if the file doesn't already have an extension)
            filename = url.split('/')[-1]
            if not filename.endswith(ext):  # To avoid doubling the extension if it already exists in the URL
                filename += ext
            filepath = folder_path / filename
        
        # Create the folder if it does not exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Check if the file already exists
        if filepath.exists():
            print(f"{filepath} already exists, skipping download..")
            return filepath
        
        # Make a request to the URL and check for errors
        print(f"Downloading {url} to {filepath}...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful
        
        # Write the content to the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def is_zip_file(file_path):
    with open(file_path, 'rb') as file:
        return file.read(4) == b'\x50\x4b\x03\x04'

def unzip_to_folder(zip_path, target_folder):
    """
    Unzip the .zip file to the target folder.
    """

    if not is_zip_file(zip_path):
        raise ValueError(f"The file {zip_path} is not a .zip file!")
    
    os.makedirs(target_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

def prep_img_dir(target_folder, extensions=['.jpg', '.png', '.jpeg', '.webm', '.JPG', '.PNG', '.JPEG', '.WEBM'], max_n_pixels = 2048*2048):
    """
    1. Move the images with given extensions to the root of target folder.
    2. Remove all subfolders.
    3. Open all images with PIL, fix their rotation, size and save them back to the same path as .jpg
    """
    try:
        print(f"Prepping image directory {target_folder}...")
        for foldername, subfolders, filenames in os.walk(target_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    source = os.path.join(foldername, filename)
                    shutil.move(source, target_folder)

        # Removing subfolders
        for foldername, subfolders, _ in os.walk(target_folder, topdown=False):  # topdown=False for bottom-up traversal
            for subfolder in subfolders:
                shutil.rmtree(os.path.join(foldername, subfolder))

        # Load all images with correct rotation and re-save as .jpg
        final_imgs = 0
        for filename in os.listdir(target_folder):
            load_path = os.path.join(target_folder, filename)
            image = eden_utils.load_image_with_orientation(load_path)

            # optionally downsizing the image:
            if image.size[0] * image.size[1] > max_n_pixels:
                image.thumbnail((2048, 2048), Image.ANTIALIAS)

            # Create save_path with .jpg extension:
            save_path = os.path.join(target_folder, os.path.splitext(filename)[0] + '.jpg')
            image.save(save_path, quality=95)
            final_imgs += 1

        print(f"Succesfully prepped {final_imgs} .jpg images in {target_folder}!")

    except Exception as e:
        print(f"An error occurred while prepping the image directory: {e}")
        print("Trying to continue anyway...")