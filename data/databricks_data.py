import requests
import base64
import os


# Define Databricks domain and token
DOMAIN = 'adb-4977039585552932.12.azuredatabricks.net'
TOKEN = 'dapief4091da62d5cb554ceb5f318e4adc60-2'  # Replace with your actual access token
DATABRICKS_URL = 'https://adb-4977039585552932.12.azuredatabricks.net'
# Set the file parameters
file_path = "dbfs:/FileStore/shared_uploads/t.jordbakke@fearnleys.com/transaction_data/"

def list_dbfs_files(path):
    result_files = []
    url = f'{DATABRICKS_URL}/api/2.0/dbfs/list'
    headers = {
        'Authorization': f'Bearer {TOKEN}'
    }

    data = {
        'path': path
    }

    response = requests.get(url, headers=headers, params=data)
    if response.status_code == 200:
        files = response.json().get('files', [])
        for file in files:
            result_files.append(file["path"])
    else:
        print(f"Error: {response.status_code}, {response.text}")
    
    return result_files

def download_dbfs_file(dbfs_file_path, local_file_path, domain, token):
    
    """
    Downloads a file from DBFS to a local path.

    Args:
    dbfs_file_path (str): The path to the file in DBFS.
    local_file_path (str): The local path where the file will be saved.
    domain (str): The Databricks domain.
    token (str): The access token for Databricks API.
    """

    # Create the directory if it doesn't exist
    local_dir = os.path.dirname(local_file_path)
    os.makedirs(local_dir, exist_ok=True)

    # Initialize parameters to read file in multiple requests
    bytes_read = 1048576  # 1 MB
    offset = 0

    # Open a file to write
    with open(local_file_path, "wb+") as f:
        
        # Keep reading the file till bytes_read is less than 1 MB
        while bytes_read >= 1048576:
            
            # API request to read the file from offset
            response = requests.get(
                f'https://{domain}/api/2.0/dbfs/read',
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'  # Ensure correct content type
                },
                json={
                    "path": dbfs_file_path,
                    "offset": offset,
                    "length": 1048576
                }
            )
            
            # Check if valid response is received
            if response.ok:
                try:
                    # Decode and write the content to file
                    bytes_read = response.json().get('bytes_read', 0)
                    encoded_bytes = response.json().get('data', '')
                    if encoded_bytes:
                        decoded_bytes = base64.b64decode(encoded_bytes)
                        f.write(decoded_bytes)
                    
                    # Increment offset for next request
                    offset += bytes_read
                    
                except Exception as exc:
                    # Exception message
                    print(f"Exception occurred. Error message: {exc}")
                    break
            else:
                # Response invalid message
                print(f"Received invalid response. Response: {response.json()}")
                break

    print(f"File downloaded successfully to {local_file_path}")

# dirs = ["fundamentals_example"]

# for path in dirs:
#     parent_path = file_path + path
#     files = list_dbfs_files(parent_path)
#     for file in files:
#         file = file.split("/")[-1]
#         if "part" in file:
            
#             download_dbfs_file(parent_path + "\\" + file,r"C:\repos\Deep-learning-trj\data" + "\\" + path + "\\" + file
#                               , DOMAIN, TOKEN)
