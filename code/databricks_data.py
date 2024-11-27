import os
import requests
import base64
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from dotenv import load_dotenv
import io

# Load the .env file from the same directory as the script
load_dotenv()

# Define your Databricks instance and token
databricks_instance = "https://adb-4977039585552932.12.azuredatabricks.net"
token = os.getenv("DATABRICKS_TOKEN")

def get_dbfs_status(path: str) -> dict:
    # Define the REST API endpoint for getting file or directory status in DBFS
    get_status_url = f"{databricks_instance}/api/2.0/dbfs/get-status"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Create the payload to specify the path
    payload = {
        "path": path
    }

    # Make the request to get the status of the file or directory
    response = requests.get(get_status_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get status from DBFS: {response.text}")

def list_dbfs_files(directory_path: str) -> list:
    # Get the status of the directory_path
    status = get_dbfs_status(directory_path)

    if status["is_dir"]:
        # If it's a directory, list the files
        list_files_url = f"{databricks_instance}/api/2.0/dbfs/list"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Create the payload to specify the directory path
        payload = {
            "path": directory_path
        }

        # Make the request to list the files in the directory
        response = requests.get(list_files_url, headers=headers, json=payload)

        if response.status_code == 200:
            files = response.json().get("files", [])
            return [f["path"] for f in files if "part" in f["path"]]  # Filter for "part" files
        else:
            raise Exception(f"Failed to list files from DBFS: {response.text}")
    else:
        # If it's a single file, return it as a list
        return [directory_path]

def read_dbfs_file(file_path: str) -> bytes:
    # Define the REST API endpoint for reading a file in DBFS
    read_file_url = f"{databricks_instance}/api/2.0/dbfs/read"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Create the payload to specify the file path
    payload = {
        "path": file_path
    }

    # Make the request to read the file
    response = requests.get(read_file_url, headers=headers, json=payload)

    if response.status_code == 200:
        file_content_base64 = response.json()["data"]
        # Decode the base64-encoded file content
        file_content = base64.b64decode(file_content_base64)
        return file_content
    else:
        raise Exception(f"Failed to read file from DBFS: {response.text}")

def load_parquet_parts_from_dbfs(directory_path: str) -> list:
    # List all part files in the directory
    part_files = list_dbfs_files(directory_path)

    # Read and combine all part files into a list of pyarrow tables
    tables = []
    for part_file in part_files:
        parquet_bytes = read_dbfs_file(part_file)
        parquet_file = pq.ParquetFile(io.BytesIO(parquet_bytes))
        tables.append(parquet_file.read())

    return tables

def write_parquet_to_local(tables: list, write_local_path: str):
    # Concatenate the pyarrow tables
    combined_table = pa.concat_tables(tables)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(write_local_path), exist_ok=True)

    # Write the combined table to a local Parquet file
    pq.write_table(combined_table, write_local_path)
    print(f"Parquet file written to {write_local_path}")



if __name__ == "__main__":
    pass