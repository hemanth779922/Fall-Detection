import cv2
from ultralytics import YOLO
import config

class PoseDetector:
    def __init__(self):
        # Initialize YOLOv8 Pose model
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.results = None
        self.img_height = 0
        self.img_width = 0

    def find_pose(self, img, draw=True):
        """
        Processes the image and extracts pose landmarks using YOLOv8.
        Returns the annotated image if draw=True, else the original image.
        """
        self.img_height, self.img_width, _ = img.shape
        
        # Run inference and tracking on the image
        # persist=True allows YOLO to track people across frames assigning them an ID
        self.results = self.model.track(img, persist=True, verbose=False)[0]
        
        if draw:
            # YOLO's plot() method automatically draws bounding boxes and skeletons
            return self.results.plot()
        return img

    def get_landmarks(self, img):
        """
        Returns a list of dictionaries for each tracked person:
        [{"id": tracking_id, "landmarks": [...]}]
        """
        tracked_people = []
        
        # Check if any person was detected and has keypoints
        if self.results and hasattr(self.results, 'keypoints') and self.results.keypoints is not None:
            keypoints_tensor = self.results.keypoints.data
            
            # Extract tracking IDs if available
            track_ids = []
            if self.results.boxes and self.results.boxes.id is not None:
                track_ids = self.results.boxes.id.int().cpu().tolist()
            else:
                # Fallback if tracking ID isn't assigned (e.g., first frame)
                track_ids = [i for i in range(len(keypoints_tensor))]

            # Loop through all detected people
            for idx, keypoints in enumerate(keypoints_tensor):
                if len(keypoints) == 0:
                    continue
                    
                track_id = track_ids[idx] if idx < len(track_ids) else idx
                landmarks_list = []
                
                for kp in keypoints:
                    px, py, conf = float(kp[0]), float(kp[1]), float(kp[2])
                    
                    # Normalize X and Y coordinates to [0, 1] range
                    norm_x = px / self.img_width if self.img_width > 0 else 0
                    norm_y = py / self.img_height if self.img_height > 0 else 0
                    
                    # We add 0 for Z coordinate
                    landmarks_list.append([norm_x, norm_y, 0.0, conf])
                    
                tracked_people.append({
                    "id": track_id,
                    "landmarks": landmarks_list
                })
                    
        return tracked_people
