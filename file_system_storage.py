import os
import glob
import pickle
from pathlib import Path

import boto3

from singleton import Singleton

BUCKET = "kcol"
DATA_PATH = f"data/"
MODEL_PATH = f"models/"
WASABI_URL = "https://s3.eu-central-1.wasabisys.com"
KEY = ""
SECERT = ""

WASABI_BASE = "https://wasabisys.com"
REGUIB = "eu-central-1"


class FS(metaclass=Singleton):
    def __init__(self):
        self.s3 = boto3.resource(
            's3',
            endpoint_url=WASABI_URL,
            aws_access_key_id=KEY,
            aws_secret_access_key=SECERT
        )
        self.boto_bucket = self.s3.Bucket(BUCKET)
        self.temp_file_mape = {}


    def upload_local_directory(self, local_path, remote_path):
        if not os.path.isdir(local_path):
            print(f"{local_path} is not a dir")
        for local_file in glob.glob(local_path + '/**'):
            if not os.path.isfile(local_file):
               self.upload_local_directory(local_file, remote_path + "/" + os.path.basename(local_file))
            else:
               remote_path = os.path.join(remote_path, local_file[1 + len(local_path):])
               self.upload_file(local_file, remote_path)

    def upload_file(self,local, remote):
        self.boto_bucket.upload_file(local, remote)

    def upload_data(self,data, remote):
        object = self.s3.Object(BUCKET, remote.replace("\\","/"))
        object.put(Body=data)
        # with open("tmp/"+remote.replace("\\","/"), "w") as f:
        #     f.write(data)


    def get_file(self, remote, local=None, download=True, override=False):
        old_sep = os.sep
        os.sep = '/'

        if local is None:
            local = os.path.normpath(os.path.join("tmp", *os.path.split(remote)))

        local_folder = os.path.dirname(local)
        os.makedirs(local_folder, exist_ok=True)

        if download and not override:
            if os.path.exists(local):
                return local

        self.boto_bucket.download_file(remote, local)

        os.sep = old_sep
        return local

    def get_data(self,remote, delimiter="", download=True, override=False):
        if remote in self.temp_file_mape:
            return self.temp_file_mape[remote]

        if not delimiter:
            delimiter = os.path.dirname(remote)

        if download:
            tmp_file = os.path.join("../tmp", remote)
            if os.path.isfile(tmp_file) and not override:
                with open(tmp_file, "rb") as f:
                    return f.read()

        body = None
        for obj in self.boto_bucket.objects.filter(Prefix =delimiter):
            key = obj.key
            if key == remote:
                body = obj.get()['Body'].read()

        if download:
            Path(os.path.dirname(tmp_file)).mkdir(parents=True, exist_ok=True)
            with open(tmp_file, "wb") as f:
                f.write(body)

        return body

    def list_items(self, prefix):
        """
        should not start with /
        """
        prefix=prefix.replace("\\", "/")
        return [obj.key for obj in self.boto_bucket.objects.filter(Prefix=prefix)]

    def file_exists(self, file):
        # file=file.replace("\\", "/")
        dir = os.path.dirname(file)
        for f in self.boto_bucket.objects.filter(Prefix=dir):
            if f.key == file:
                return True
        return False

    def folder_exists(self, file):
        file=file.replace("\\", "/")
        return len(list(self.boto_bucket.objects.filter(Prefix=file))) > 0

    def add_temp_file(self, path, content):
        self.temp_file_mape[path] = content
