import cv2
import threading
import time


class CameraStream:
    def __init__(self, camera_index, camera_name):
        self.camera_index = camera_index
        self.camera_name = camera_name

        # CAP_DSHOW usually improves USB camera startup on Windows
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # Start with moderate settings for 3-camera testing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.cap.isOpened():
            print(
                f"ERROR: Could not open {self.camera_name} "
                f"at index {self.camera_index}"
            )
            return False

        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

        print(
            f"Opened {self.camera_name} "
            f"at index {self.camera_index}"
        )
        return True

    def update(self):
        while self.running:
            success, frame = self.cap.read()

            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.cap.isOpened():
            self.cap.release()


# Change these indexes after checking your camera numbering
CAMERA_CONFIG = [
    {"index": 0, "name": "Camera 1"},
    {"index": 1, "name": "Camera 2"},
    {"index": 2, "name": "Camera 3"},
]

cameras = []

try:
    # Start all cameras
    for config in CAMERA_CONFIG:
        camera = CameraStream(
            camera_index=config["index"],
            camera_name=config["name"]
        )

        if camera.start():
            cameras.append(camera)

    if not cameras:
        print("No cameras could be opened.")
        raise SystemExit

    print("\nPress Q to close all camera windows.\n")

    while True:
        for camera in cameras:
            frame = camera.get_frame()

            if frame is None:
                continue

            # Display camera name and index
            label = (
                f"{camera.camera_name} | "
                f"Index: {camera.camera_index}"
            )

            cv2.putText(
                frame,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(camera.camera_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    for camera in cameras:
        camera.stop()

    cv2.destroyAllWindows()
    print("All cameras stopped.")
