import math

class FeatureExtractor:
    def __init__(self):
        self.prev_hip_y = None
        self.prev_hip_x = None
        self.prev_shoulder_x = None
        self.prev_shoulder_y = None
        self.prev_nose_y = None
        self.head_speed = 0.0

    def extract_features(self, landmarks):
        """
        Converts full body landmarks (33 points) into 4 specific Machine Learning features.
        Returns a 1D list of features or None if landmarks are incomplete.
        """
        if not landmarks or len(landmarks) < 13:
            return None

        left_shoulder = landmarks[5]
        right_shoulder = landmarks[6]
        left_hip = landmarks[11]
        right_hip = landmarks[12]

        # Calculate midpoints for better stability instead of using just one side
        mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        mid_hip_x = (left_hip[0] + right_hip[0]) / 2
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # --- Feature 1: Body Tilt Angle (Degrees) ---
        # Calculate the angle between the body axis (from hip to shoulder) and the vertical axis
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_shoulder_y - mid_hip_y
        # math.atan2 computes the angle; we convert it to degrees
        # A standing person will have an angle close to 0 (or 180).
        # A person lying down will have an angle close to 90 degrees.
        tilt_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        
        # --- Feature 2: Height/Width Ratio of Bounding Box ---
        # Compute the bounding box encompassing the person's landmarks
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # Ratio will be high (~3.0) when standing, low (<1.0) when lying down
        hw_ratio = height / (width + 1e-6)
        
        # --- Feature 3: Vertical Position of Hip (Normalized Y) ---
        # Coordinate Y is 0 at the top edge of screen, 1 at the bottom.
        # When a person falls, their hip goes closer to the floor (closer to >0.8)
        vertical_hip_pos = mid_hip_y
        
        # --- Feature 4: Speed of Movement (Hip Y Drop Speed) ---
        # Calculate how fast the hip is moving downwards between frames
        hip_speed = 0
        if self.prev_hip_y is not None:
            # Positive value means the person is dropping downwards
            hip_speed = mid_hip_y - self.prev_hip_y  
        
        # --- Feature 5: Total Body Movement (Euclidean Distance) ---
        # Useful for detecting if a person is tossing/turning while sleeping
        total_movement = 0.0
        if (self.prev_hip_x is not None and self.prev_hip_y is not None and 
            self.prev_shoulder_x is not None and self.prev_shoulder_y is not None):
            
            # Distance moved by hips
            hip_dist = math.sqrt((mid_hip_x - self.prev_hip_x)**2 + (mid_hip_y - self.prev_hip_y)**2)
            # Distance moved by shoulders
            shoulder_dist = math.sqrt((mid_shoulder_x - self.prev_shoulder_x)**2 + (mid_shoulder_y - self.prev_shoulder_y)**2)
            
            total_movement = hip_dist + shoulder_dist

        # Update previous frame memory
        self.prev_hip_x = mid_hip_x
        self.prev_hip_y = mid_hip_y
        self.prev_shoulder_x = mid_shoulder_x
        self.prev_shoulder_y = mid_shoulder_y

        # Track head (nose) speed for sudden fall overrides
        nose = landmarks[0]
        nose_y = nose[1]
        
        self.head_speed = 0.0
        if self.prev_nose_y is not None:
            self.head_speed = nose_y - self.prev_nose_y
            
        self.prev_nose_y = nose_y

        # Return the extracted features as a list (Now 5 elements)
        return [tilt_angle, hw_ratio, vertical_hip_pos, hip_speed, total_movement]
