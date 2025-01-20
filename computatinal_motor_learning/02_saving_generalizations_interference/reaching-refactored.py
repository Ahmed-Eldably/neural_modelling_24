import pygame
import random
import math
import numpy as np
import pandas as pd
import os

# ========================== EXPERIMENT CONFIGURATION ==========================
class Config:
    SCREEN_X, SCREEN_Y = 3840, 2160  # Screen resolution
    WIDTH, HEIGHT = SCREEN_X // 1.5, SCREEN_Y // 1.5  # Adjust for scaling
    CIRCLE_SIZE = 20
    TARGET_SIZE = CIRCLE_SIZE
    TARGET_RADIUS = 300
    MASK_RADIUS = 0.66 * TARGET_RADIUS
    ATTEMPTS_LIMIT = 160
    TIME_LIMIT = 1000  # Time limit per attempt (ms)

    # Start Positions
    START_POSITION = (WIDTH // 2, HEIGHT // 2)
    START_ANGLE = 0
    PERTURBATION_ANGLE = math.radians(30)  # Perturbation Angle in Radians

    # Colors
    WHITE, BLACK, RED, BLUE = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255)

# ========================== PARTICIPANT DATA HANDLING ==========================
class Participant:
    def __init__(self, participant_id, age, gender, handedness):
        self.id = participant_id
        self.age = age
        self.gender = gender
        self.handedness = handedness
        self.session = self.get_next_session_number()
        self.trial_data = []  # Stores all trials

    def get_next_session_number(self):
        """Determine the next session number for the participant to avoid overwriting."""
        session_number = 1
        while os.path.exists(f'participant_{self.id}_session_{session_number}.csv'):
            session_number += 1
        return session_number

    def log_trial(self, attempt, error_angle, reaction_time, movement_duration, hit):
        """Store each trial's results."""
        self.trial_data.append({
            "Participant ID": self.id,
            "Session": self.session,
            "Attempt": attempt,
            "Error Angle": error_angle,
            "Reaction Time (ms)": reaction_time,
            "Movement Duration (ms)": movement_duration,
            "Hit": hit
        })

    def save_results(self):
        """Save participant's session data to CSV."""
        df = pd.DataFrame(self.trial_data)
        filename = f'participant_{self.id}_session_{self.session}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved for Participant {self.id}, Session {self.session} in {filename}")

# ========================== EXPERIMENT CLASS ==========================
class MotorLearningExperiment:
    def __init__(self, participant):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Reaching Game")
        self.clock = pygame.time.Clock()

        # Game State Variables
        self.score, self.attempts = 0, 0
        self.new_target, self.start_time = None, 0
        self.move_faster = False

        # Perturbation Variables
        self.perturbation_mode = True
        self.perturbation_type = 'gradual'  # 'random', 'gradual', 'sudden'
        self.gradual_step, self.gradual_attempts, self.prev_gradual_attempts = 0, 1, 0
        self.perturbation_rand = random.uniform(-math.pi / 4, +math.pi / 4)

        # Experiment Settings
        self.target_mode = 'fix'  # 'random' or 'fix'
        self.show_mouse_info = False

        # Participant Data
        self.participant = participant
        self.trial_start_time = 0  # Track when each trial starts

        # Ensure the first target is generated
        self.new_target = self.generate_target_position()

    # ========================== HELPER FUNCTIONS ==========================
    def generate_target_position(self):
        """Generate a new target position based on the selected target mode."""
        angle = random.uniform(0, 2 * math.pi) if self.target_mode == 'random' else math.radians(Config.START_ANGLE)
        return [
            Config.WIDTH // 2 + Config.TARGET_RADIUS * math.sin(angle),
            Config.HEIGHT // 2 + Config.TARGET_RADIUS * -math.cos(angle)
        ]

    def compute_error_angle(self, start_position, target, circle_pos):
        """Compute angular error between start, target, and movement endpoint."""
        if target is None or circle_pos is None:
            return None  # Avoid computation errors

        s, t, c = np.array(start_position), np.array(target), np.array(circle_pos)
        A, B = t - s, c - s
        det = A[0] * B[1] - A[1] * B[0]
        return np.degrees(np.arctan2(det, np.dot(A, B)))  # Clockwise error is negative

    def update_perturbation(self, mouse_angle, distance):
        """Calculate perturbed mouse position based on perturbation type (same as original code)."""
        if self.perturbation_mode:
            if self.perturbation_type == 'sudden':
                perturbed_mouse_angle = mouse_angle + Config.PERTURBATION_ANGLE
            elif self.perturbation_type == 'gradual':
                perturbed_mouse_angle = mouse_angle - self.gradual_step * (Config.PERTURBATION_ANGLE / 10)

                if self.gradual_attempts % 3 == 0 and self.prev_gradual_attempts != self.gradual_attempts:
                    self.gradual_step += 1

                if self.gradual_step >= 10:
                    self.gradual_attempts = 0  # Reset after 10 steps for cyclic perturbation

            return [
                Config.START_POSITION[0] + distance * math.cos(perturbed_mouse_angle),
                Config.START_POSITION[1] + distance * math.sin(perturbed_mouse_angle)
            ]

        return pygame.mouse.get_pos()  # No perturbation

    # ========================== MAIN GAME LOOP ==========================
    def run(self):
        """Main game loop - identical in functionality to the original version."""
        running = True

        while running:
            self.screen.fill(Config.BLACK)

            # ========================== EVENT HANDLING ==========================
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_4:
                        self.perturbation_mode = True
                    elif event.key == pygame.K_5:
                        self.perturbation_mode = False
                    elif event.key == pygame.K_h:
                        self.show_mouse_info = not self.show_mouse_info

            # ========================== EXPERIMENT DESIGN ==========================
            if self.attempts == 1:
                self.perturbation_mode = False
            elif self.attempts == 20:
                self.perturbation_mode = True
                self.perturbation_type = 'sudden'
            elif self.attempts == 60:
                self.perturbation_mode = False
            elif self.attempts == 80:
                self.perturbation_mode = True
                self.perturbation_type = 'sudden'
            elif self.attempts == 140:
                self.perturbation_mode = False
            elif self.attempts >= Config.ATTEMPTS_LIMIT:
                running = False

            # Hide the mouse cursor
            pygame.mouse.set_visible(False)

            # ========================== TRACK MOUSE MOVEMENT ==========================
            mouse_pos = pygame.mouse.get_pos()
            deltax, deltay = mouse_pos[0] - Config.START_POSITION[0], mouse_pos[1] - Config.START_POSITION[1]
            distance, mouse_angle = math.hypot(deltax, deltay), math.atan2(deltay, deltax)

            # ========================== CALCULATE PERTURBATION ==========================
            if self.perturbation_mode:
                if self.perturbation_type == 'sudden':
                    perturbed_mouse_angle = mouse_angle + Config.PERTURBATION_ANGLE
                elif self.perturbation_type == 'gradual':
                    perturbed_mouse_angle = mouse_angle - self.gradual_step * (Config.PERTURBATION_ANGLE / 10)

                    if self.gradual_attempts % 3 == 0 and self.prev_gradual_attempts != self.gradual_attempts:
                        self.gradual_step += 1

                    if self.gradual_step >= 10:
                        self.gradual_attempts = 0  # Reset after 10 steps for cyclic perturbation

                circle_pos = [
                    Config.START_POSITION[0] + distance * math.cos(perturbed_mouse_angle),
                    Config.START_POSITION[1] + distance * math.sin(perturbed_mouse_angle)
                ]
            else:
                circle_pos = pygame.mouse.get_pos()

            # ========================== CHECK TARGET HIT OR MISS ==========================
            hit = False
            reaction_time = pygame.time.get_ticks() - self.trial_start_time
            movement_duration = pygame.time.get_ticks() - self.start_time

            if self.new_target and math.hypot(circle_pos[0] - self.new_target[0],
                                              circle_pos[1] - self.new_target[1]) <= Config.CIRCLE_SIZE:
                self.score += 1
                self.attempts += 1
                hit = True

                error_angle = self.compute_error_angle(Config.START_POSITION, self.new_target, circle_pos)
                if error_angle is not None:
                    self.participant.log_trial(self.attempts, error_angle, reaction_time, movement_duration, hit)

                self.new_target = None  # Reset target
                self.start_time = 0  # Reset timer
                if self.perturbation_type == 'gradual' and self.perturbation_mode:
                    self.gradual_attempts += 1

            elif self.new_target and math.hypot(circle_pos[0] - Config.START_POSITION[0],
                                                circle_pos[1] - Config.START_POSITION[1]) > Config.TARGET_RADIUS * 1.01:
                self.attempts += 1
                hit = False

                error_angle = self.compute_error_angle(Config.START_POSITION, self.new_target, circle_pos)
                if error_angle is not None:
                    self.participant.log_trial(self.attempts, error_angle, reaction_time, movement_duration, hit)

                self.new_target = None  # Reset target
                self.start_time = 0  # Reset timer

                if self.perturbation_type == 'gradual' and self.perturbation_mode:
                    self.gradual_attempts += 1

            # ========================== CHECK IF NEW TARGET IS NEEDED ==========================
            if not self.new_target and math.hypot(mouse_pos[0] - Config.START_POSITION[0],
                                                  mouse_pos[1] - Config.START_POSITION[1]) <= Config.CIRCLE_SIZE:
                self.new_target = self.generate_target_position()
                self.start_time = pygame.time.get_ticks()

            # ========================== TIME LIMIT CHECK ==========================
            if self.start_time and (pygame.time.get_ticks() - self.start_time) > Config.TIME_LIMIT:
                self.move_faster = True
                self.start_time = 0  # Reset timer

            # ========================== DISPLAY 'MOVE FASTER' WARNING ==========================
            if self.move_faster:
                font = pygame.font.Font(None, 36)
                text = font.render('MOVE FASTER!', True, Config.RED)
                text_rect = text.get_rect(center=(Config.START_POSITION))
                self.screen.blit(text, text_rect)

            # ========================== DRAW GAME ELEMENTS ==========================
            if self.new_target:
                pygame.draw.circle(self.screen, Config.BLUE, self.new_target, Config.TARGET_SIZE // 2)

            pygame.draw.circle(self.screen, Config.WHITE, circle_pos, Config.CIRCLE_SIZE // 2)
            pygame.draw.circle(self.screen, Config.WHITE, Config.START_POSITION, 5)

            # ========================== DISPLAY SCORE & ATTEMPTS ==========================
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, Config.WHITE)
            self.screen.blit(score_text, (10, 10))

            attempts_text = font.render(f"Attempts: {self.attempts}", True, Config.WHITE)
            self.screen.blit(attempts_text, (10, 30))

            # ========================== DEBUGGING INFO (IF TOGGLED) ==========================
            if self.show_mouse_info:
                mouse_info_text = font.render(f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, Config.WHITE)
                delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, Config.WHITE)
                mouse_angle_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, Config.WHITE)
                self.screen.blit(mouse_info_text, (10, 60))
                self.screen.blit(delta_info_text, (10, 90))
                self.screen.blit(mouse_angle_text, (10, 120))

            # ========================== UPDATE DISPLAY ==========================
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        self.participant.save_results()

# ========================== RUN EXPERIMENT ==========================
if __name__ == "__main__":
    participant = Participant(participant_id=1, age=25, gender="M", handedness="Right")
    experiment = MotorLearningExperiment(participant)
    experiment.run()
