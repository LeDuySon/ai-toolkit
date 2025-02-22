import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import glob
from typing import Union, OrderedDict
from dotenv import load_dotenv
import boto3

# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ["DISABLE_TELEMETRY"] = "YES"

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch

    torch.autograd.set_detect_anomaly(True)

import argparse
from toolkit.job import get_job

print(f"Initializing S3 client connect to {os.environ.get('SAVE_CHECKPOINT_BUCKET_NAME')}")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("SAVE_CHECKPOINT_AWS_REGION"),
)

def print_end_message(jobs_completed, jobs_failed):
    failure_string = (
        f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}"
        if jobs_failed > 0
        else ""
    )
    completed_string = (
        f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    )

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


def upload_to_s3(folder_path: str, bucket_name: str, pattern: str = ".safetensors"):
    files = glob.glob(os.path.join(folder_path, f"*{pattern}"))
    object_path = f"{os.environ.get('USER_ID')}/{os.environ.get('JOB_ID')}/lora_checkpoints"
    
    print(f"Uploading {len(files)} files to {bucket_name}/{object_path}")
    for file in files:
        print(
            f"Uploading {file} to {bucket_name}/{object_path}"
        )
        s3_client.upload_file(
            file,
            Bucket=bucket_name,
            Key=os.path.join(object_path, os.path.basename(file)),
        )
        
def clear_target_folder(folder_path: str):
    import shutil 
    
    # delete all files and folders in the workspace
    for part in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, part)):
            print(f"Deleting file: {os.path.join(folder_path, part)}")
            os.remove(os.path.join(folder_path, part))
        elif os.path.isdir(os.path.join(folder_path, part)):
            print(f"Deleting folder: {os.path.join(folder_path, part)}")
            shutil.rmtree(os.path.join(folder_path, part))


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        "config_file_list",
        nargs="+",
        type=str,
        help="Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially",
    )

    # flag to continue if failed job
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        help="Continue running additional jobs even if a job fails",
    )

    # flag to continue if failed job
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name to replace [name] tag in config file, useful for shared config file",
    )
    parser.add_argument(
        "--shutdown", action="store_true", help="Automatically shut down when done"
    )
    parser.add_argument(
        "--shutdown-time",
        "-st",
        type=int,
        default=120,
        help="Time to wait before automatically shutting down, in seconds.",
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        default="/workspace",
        help="Target folder to save the output, if not provided, the default will be /workspace",
    )
    args = parser.parse_args()

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    print(
        f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}"
    )

    for config_file in config_file_list:
        try:
            print(f"Running job: {config_file}")
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1

            print(f"Job completed: {config_file}")
            print(f"Output directory: {job.config}")
            if os.environ.get("SAVE_CHECKPOINT_BUCKET_NAME") is not None:
                bucket_name = os.environ.get("SAVE_CHECKPOINT_BUCKET_NAME")
                output_folder = job.config["process"][0]["training_folder"]
                checkpoint_dir = os.path.join(output_folder, job.config["name"])
                upload_to_s3(folder_path=checkpoint_dir, bucket_name=bucket_name)
                
                print(f"Clearing target folder: {checkpoint_dir}")
                clear_target_folder(args.target_folder)
                
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    if args.shutdown:
        import time
        import subprocess

        pod_id = os.environ.get("RUNPOD_POD_ID", None)
        if pod_id is not None:
            print(
                f"Automatic shut down is configured. Shutting down in {args.shutdown_time} seconds! Hit Control-C to cancel."
            )
            try:
                time.sleep(args.shutdown_time)
                subprocess.run(f"runpodctl stop pod {pod_id}", shell=True, check=False)
            except KeyboardInterrupt:
                print("Automatic shut down cancelled.")
        else:
            print(
                "Automatic shut down was configured, but could not get environment $RUNPOD_POD_ID"
            )


if __name__ == "__main__":
    main()
