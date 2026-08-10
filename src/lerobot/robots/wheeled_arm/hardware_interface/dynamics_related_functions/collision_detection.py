class Collision_Detection:
    """Minimal MOVEJ-compatible collision detection interface."""

    def __init__(self, lcm_handler):
        self.lcm_handler = lcm_handler
        self.collision_detection_level = 0
        self.collision_detection_index = False

    def start_collision_detection(self):
        self.collision_detection_index = False

    def stop_collision_detection(self):
        self.collision_detection_index = False
