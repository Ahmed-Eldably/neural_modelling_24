import math
import os




class TaskConfig:
    SCREEN_X = int(os.getenv('SCREEN_X', 3840))  # Screen resolution X
    SCREEN_Y = int(os.getenv('SCREEN_Y', 2160))  # Screen resolution Y
    WIDTH, HEIGHT = SCREEN_X // 1.5, SCREEN_Y // 1.5  # Adjust for scaling
    CIRCLE_SIZE = 20
    TARGET_SIZE = CIRCLE_SIZE
    TARGET_RADIUS = 300
    MASK_RADIUS = 0.66 * TARGET_RADIUS
    ATTEMPTS_LIMIT = 200
    TIME_LIMIT = 1000  # Time limit per attempt (ms)

    # Start Positions
    START_POSITION = (WIDTH // 2, HEIGHT // 2)
    PERTURBATION_ANGLE = math.radians(30)  # Perturbation Angle in Radians

    # Colors
    WHITE, BLACK, RED, BLUE = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255)
    def __init__(self, experiment_name, phase):
        self.experiment_name = experiment_name
        self.phase = phase

        if experiment_name == "savings":
            TaskConfig.START_ANGLE = 0
            if phase == 0:
                pass    # First time doing the savings
            elif phase == 1:
                pass  # Second time doing the savings after 30 mins
        elif experiment_name == "generalization":
            if phase == 0:
                TaskConfig.START_ANGLE = 20
            if phase == 1:
                TaskConfig.START_ANGLE = 50
            elif phase == 2:
                TaskConfig.START_ANGLE = 35
            elif phase == 3:
                TaskConfig.START_ANGLE = 125



