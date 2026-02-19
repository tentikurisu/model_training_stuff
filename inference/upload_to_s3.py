"""
S3 Upload Script
================
Upload cropped images to S3 for OCR processing.

Run this AFTER Cell 1 has generated the crops.
"""

import os
import boto3
from pathlib import Path
import yaml


# ============================================================
# CELL 2: S3 UPLOAD
# ============================================================


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def upload_to_s3(config=None):
    """Upload cropped images to S3"""

    # Load config
    if config is None:
        config = load_config()

    print("=" * 60)
    print("S3 UPLOAD")
    print("=" * 60)

    # Get S3 settings
    s3_bucket = config.get("s3_bucket", "popup-detector-uploads")
    s3_prefix = config.get("s3_prefix", "ocr_queue/")
    aws_region = config.get("aws_region", "us-east-1")
    output_folder = config.get("output_folder", "output/crops")

    print(f"\nS3 Bucket: {s3_bucket}")
    print(f"S3 Prefix: {s3_prefix}")
    print(f"Region: {aws_region}")
    print(f"Source folder: {output_folder}")

    # Initialize S3 client
    print("\nInitializing S3 client...")
    s3_client = boto3.client("s3", region_name=aws_region)

    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    files_to_upload = []

    for ext in image_extensions:
        files_to_upload.extend(Path(output_folder).glob(f"*{ext}"))

    if not files_to_upload:
        print(f"\nERROR: No files found in {output_folder}")
        print("Run Cell 1 first to generate crops!")
        return

    print(f"\nFound {len(files_to_upload)} files to upload")

    # Upload files
    uploaded = 0
    failed = 0

    for file_path in files_to_upload:
        try:
            s3_key = f"{s3_prefix}{file_path.name}"

            s3_client.upload_file(
                str(file_path),
                s3_bucket,
                s3_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )

            uploaded += 1
            if uploaded % 10 == 0:
                print(f"  Uploaded {uploaded}/{len(files_to_upload)}...")

        except Exception as e:
            failed += 1
            print(f"  ERROR uploading {file_path.name}: {e}")

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Successfully uploaded: {uploaded}")
    print(f"Failed: {failed}")
    print(f"S3 Location: s3://{s3_bucket}/{s3_prefix}")
    print("\nPartner can now process with OCR!")


# Run Cell 2
if __name__ == "__main__":
    upload_to_s3()
