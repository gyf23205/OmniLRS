__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Nav-cam downlink for the perseverance rover.

A much smaller relative of PragyaanCameraHandler: one camera, one view type, one bucket. It sits on
top of ImagesHandler the same way - snapping and encoding here, bucket and parameter bookkeeping
there - so the two rovers share the Yamcs plumbing rather than reimplementing it.

Images are the one thing on a real low-rate link that genuinely cannot be afforded at telemetry
rates, so this runs on its own slow interval rather than with the 1 Hz parameter downlink.

Two fault paths meet here, deliberately at different levels:
  - frame LOSS is a downlink decision, taken here, and the frame is simply never sent
  - frame NOISE is a sensor property, applied inside Robot.get_rgba_camera_view
A blind camera and a grainy one look nothing alike from the ground, and separating them keeps it
that way.
"""

import numpy as np
from PIL import Image


class PerseveranceCameraHandler:
    """Captures the nav-cam view and hands it to the images handler for downlink."""

    def __init__(self, images_handler, robot, camera_conf, fault_injector=None):
        self._images_handler = images_handler
        self._robot = robot
        self._bucket = camera_conf.get("bucket", "images_navcam")
        self._resolution = camera_conf.get("resolution", "low")
        self._faults = fault_injector
        self._sent = 0

    def transmit_camera_view(self):
        """
        Downlink one nav-cam frame. Registered as a repeating interval by PerseveranceController.

        Errors are caught rather than raised: this runs on Kit's update stream, where an escaping
        exception takes down the subscription and with it the simulation loop.
        """
        if self._faults is not None and self._faults.should_drop_camera_frame():
            print("[cam] frame lost to an injected camera fault", flush=True)
            return

        try:
            frame = self._robot.get_rgba_camera_view(self._resolution)
        except Exception as exc:
            print(f"[cam] capture failed: {exc}", flush=True)
            return

        # The camera returns an empty array until it has rendered at least once, which the first
        # few intervals will hit while the scene is still loading.
        if getattr(frame, "size", 0) == 0:
            return

        image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGBA")
        self._images_handler.save_image(image, self._bucket)
        self._sent += 1

    @property
    def frames_sent(self) -> int:
        return self._sent
