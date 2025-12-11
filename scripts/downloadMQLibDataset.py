import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import sys

# --- Script Configuration ---
BUCKET_NAME = "mqlibinstances"
MAX_RETRIES = 10

if len(sys.argv) != 2:
    print("Usage: python downloadMQLibDataset.py outputFolder")
    exit(1)

output_folder = sys.argv[1]

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Create a boto3 client with anonymous (unsigned) access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Use a paginator to handle listing all objects in the bucket,
# even if there are more than 1000.
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=BUCKET_NAME)

print(f"Starting download of graphs from S3 bucket '{BUCKET_NAME}'...")

for page in pages:
    if 'Contents' not in page:
        continue # Skip empty pages
    
    for k in page['Contents']:
        name = k['Key']
        fname = os.path.join(output_folder, name)
        
        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if os.path.isfile(fname):
            print(f"[ Skipping {name} ]")
            continue

        print(f"Downloading {name} ...")
        
        success = False
        for count in range(MAX_RETRIES):
            try:
                s3.download_file(BUCKET_NAME, name, fname)
                success = True
                break  # Exit retry loop on success
            except Exception as e:
                print(f"  Download attempt {count + 1}/{MAX_RETRIES} failed: {e}")

        if not success:
            print(f"Failed to download {name} after {MAX_RETRIES} attempts.")
            exit(1)

print("\nAll graphs downloaded successfully.")