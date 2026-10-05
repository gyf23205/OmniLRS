__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

import omni.kit.app
import os
from PIL import Image

class ImagesHandler:
    
    NO_DATA_RESOLUTION = (640, 480)

    #NOTE
    # image_0 is no no_data image
    # image_1 is the first image with real camera view
    INIT_COUNT = 0 

    def __init__(self, yamcs_processor, yamcs_address, images_conf, url_full_nginx,
                 storage_client=None, instance=None):
        self._yamcs_processor = yamcs_processor
        self._yamcs_address = yamcs_address
        self.URL_FULL_NGINX = url_full_nginx
        self._NO_DATA_IMAGE_PATH = images_conf["no_data_image_path"]
        # Optional: with a storage client, saved images are uploaded into the Yamcs bucket so the
        # urls published below actually resolve. Without one the behaviour is what it always was -
        # write the file locally and publish where it would live - which is what pragyaan expects.
        self._storage_client = storage_client
        self._instance = instance
        self._bucket_ready = {}
        self._buckets = {}
        self._counter = {}
        self._init_buckets_and_counter(images_conf["buckets"])

    def set_storage_client(self, storage_client, instance=None):
        """
        Enable uploading saved images into the Yamcs bucket. See _upload_to_yamcs.

        instance is accepted and recorded for callers that pass it, but the storage api is
        server-global - buckets are not per-instance - so it is not sent with the requests.
        """
        self._storage_client = storage_client
        self._instance = instance
        self._bucket_ready = {}

    def _init_buckets_and_counter(self, buckets_conf):
        for bucket in buckets_conf:
            self.add_bucket(bucket["name"], bucket["path"])

    def add_bucket(self, bucket_name, path, init_count=INIT_COUNT):
        if bucket_name not in self._buckets:
            self._init_bucket(bucket_name, path, init_count)
        else:
            raise Exception(f"Bucket with name {bucket_name} already exists.")
        
    def _init_bucket(self, bucket_name, path, init_count):
        self._buckets[bucket_name] = path
        self._counter[bucket_name] = init_count

    def snap_no_data_images(self):
        for bucket_name in self._buckets:
            self.snap_no_data_image(bucket_name)

    def snap_no_data_image(self, bucket_name):
        image_path = self._NO_DATA_IMAGE_PATH 
        img = Image.open(image_path).convert('RGB').resize((self.NO_DATA_RESOLUTION[0], self.NO_DATA_RESOLUTION[1]))
        self.save_image(img, bucket_name)

    def save_image(self, image:Image, bucket_name:str):
        image_name = self._save_image_locally(image, bucket_name)
        self._upload_to_yamcs(image_name, bucket_name)
        self._inform_yamcs(image_name, bucket_name)
        self._counter[bucket_name] += 1

    def _upload_to_yamcs(self, image_name, bucket):
        """
        Put the image in the Yamcs bucket, creating the bucket on first use.

        Uploading rather than sharing a directory means the two containers need not agree about
        /tmp, and nothing depends on how Yamcs resolves a file-backed bucket path. Failure is
        logged, never raised: a ground station that will not take an image must not stop the rover.
        """
        if self._storage_client is None:
            return

        try:
            # The bucket is global to the server, so none of these take an instance. Checked once:
            # create_bucket on an existing bucket is an error, and this runs on every frame.
            if not self._bucket_ready.get(bucket):
                if bucket not in {b.name for b in self._storage_client.list_buckets()}:
                    self._storage_client.create_bucket(bucket)
                self._bucket_ready[bucket] = True

            with open(f"/tmp/{bucket}/{image_name}", "rb") as handle:
                self._storage_client.upload_object(bucket, image_name, handle,
                                                   content_type="image/png")
        except Exception as exc:
            print(f"[gs] image upload failed ({bucket}/{image_name}): {exc}", flush=True)

    def _save_image_locally(self, image, bucket) -> str:
        counter_number = self._counter[bucket]
        image_name = f"{bucket}_{counter_number:04d}.png"
        IMG_DIR = f"/tmp/{bucket}"
        os.makedirs(IMG_DIR, exist_ok=True)   # creates directory if missing
        img_path = f"{IMG_DIR}/{image_name}" 
        image.save(img_path)

        return image_name

    def _inform_yamcs(self, image_name, bucket):
        counter_number = self._counter[bucket]
        url_storage = f"/storage/buckets/{bucket}/objects/{image_name}"
        url_full = "http://" + self._yamcs_address + f"/api{url_storage}"
        url_full_nginx = self.URL_FULL_NGINX + f"/api{url_storage}" 
        self._yamcs_processor.set_parameter_values({
            self._buckets[bucket] + "/number": counter_number,
            self._buckets[bucket] + "/name": image_name,
            self._buckets[bucket] + "/url_storage": url_storage,
            self._buckets[bucket] + "/url_full": url_full,
            self._buckets[bucket] + "/url_full_nginx": url_full_nginx,
        })