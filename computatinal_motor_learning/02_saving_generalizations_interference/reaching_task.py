import argparse

import pygame
import random
import math
import numpy as np
import pandas as pd
import os
from task_config import TaskConfig


# ========================== EXPERIMENT CONFIGURATION ==========================
# class self.config:
#     SCREEN_X = int(os.getenv('SCREEN_X', 3840))  # Screen resolution X
#     SCREEN_Y = int(os.getenv('SCREEN_Y', 2160))  # Screen resolution Y
#     WIDTH, HEIGHT = SCREEN_X // 1.5, SCREEN_Y // 1.5  # Adjust for scaling
#     CIRCLE_SIZE = 20
#     TARGET_SIZE = CIRCLE_SIZE
#     TARGET_RADIUS = 300
#     MASK_RADIUS = 0.66 * TARGET_RADIUS
#     ATTEMPTS_LIMIT = 200
#     TIME_LIMIT = 1000  # Time limit per attempt (ms)
#
#     # Start Positions
#     START_POSITION = (WIDTH // 2, HEIGHT // 2)
#     START_ANGLE = 0
#     PERTURBATION_ANGLE = math.radians(30)  # Perturbation Angle in Radians
#
#     # Colors
#     WHITE, BLACK, RED, BLUE = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255)

# ========================== PARTICIPANT DATA HANDLING ==========================
class Participant:
    def __init__(self, participant_id, age, gender, handedness, task_config):
        self.id = participant_id
        self.age = age
        self.gender = gender
        self.handedness = handedness
        self.session = self.get_next_session_number()
        self.trial_data = []  # Stores all trials
        self.task_config = task_config

    def get_next_session_number(self):
        """Determine the next session number for the participant to avoid overwriting."""
        session_number = 1
        while os.path.exists(f'participant_data/{self.id}/participant_{self.id}_session_{session_number}.csv'):
            session_number += 1
        return session_number

    def log_trial(self, attempt, error_angle, hit):
        """Store each trial's results."""
        self.trial_data.append({
            "Participant ID": self.id,
            "Session": self.session,
            "Attempt": attempt,
            "Error Angle": round(error_angle, 2),
            "Hit": hit
        })

    def save_results(self):
        """Save participant's session data to CSV."""
        df = pd.DataFrame(self.trial_data)

        # Define directory and filename
        directory_location = f'participant_data/{self.id}/{self.task_config.experiment_name}/{self.task_config.phase}'
        directory = os.path.join(os.path.dirname(__file__), f'{directory_location}')
        filename = f'{directory}/participant_{self.id}_session_{self.session}.csv'

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Save the file
        df.to_csv(filename, index=False)
        print(f"Data saved for Participant {self.id}, Session {self.session} in {filename}")


# ========================== EXPERIMENT CLASS ==========================
class MotorLearningExperiment:
    def __init__(self, participant, task_config):
        self.config = task_config
        self.error_angles = []
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT), pygame.FULLSCREEN)
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
        self.mask_mode = True

        # Participant Data
        self.participant = participant
        self.trial_start_time = 0  # Track when each trial starts

        # Ensure the first target is generated
        self.new_target = self.generate_target_position()

    # ========================== HELPER FUNCTIONS ==========================
    def generate_target_position(self):
        """Generate a new target position based on the selected target mode."""
        angle = random.uniform(0, 2 * math.pi) if self.target_mode == 'random' else math.radians(self.config.START_ANGLE)
        return [
            self.config.WIDTH // 2 + self.config.TARGET_RADIUS * math.sin(angle),
            self.config.HEIGHT // 2 + self.config.TARGET_RADIUS * -math.cos(angle)
        ]

    def update_experiment_design(self):
        """Updates the perturbation mode and experiment state based on the current attempt."""
        perturbation_changes = {
            1: (False, None),  # Turn OFF perturbation
            20: (True, "sudden"),  # Sudden perturbation ON
            80: (False, None),  # Turn OFF perturbation
            100: (False, None),  # Turn OFF perturbation
            120: (True, "sudden"),  # Sudden perturbation ON
            180: (False, None),  # Turn OFF perturbation
            200: (False, None)  # Turn OFF perturbation
        }

        # Update perturbation mode if attempt matches the keys
        if self.attempts in perturbation_changes:
            self.perturbation_mode, self.perturbation_type = perturbation_changes[self.attempts]

    def check_target_reached(self, circle_pos):
        if self.new_target:
            distance = math.hypot(circle_pos[0] - self.new_target[0],
                                             circle_pos[1] - self.new_target[1])
            return distance <= self.config.CIRCLE_SIZE
        return False

    def at_start_position_and_generate_target(self, mouse_pos):
        distance = math.hypot(mouse_pos[0] - self.config.START_POSITION[0], mouse_pos[1] - self.config.START_POSITION[1])
        if distance <= self.config.CIRCLE_SIZE:
            return True
        return False

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
                perturbed_mouse_angle = mouse_angle + self.config.PERTURBATION_ANGLE
            elif self.perturbation_type == 'gradual':
                perturbed_mouse_angle = mouse_angle - self.gradual_step * (self.config.PERTURBATION_ANGLE / 10)

                if self.gradual_attempts % 3 == 0 and self.prev_gradual_attempts != self.gradual_attempts:
                    self.gradual_step += 1

                if self.gradual_step >= 10:
                    self.gradual_attempts = 0  # Reset after 10 steps for cyclic perturbation

            return [
                self.config.START_POSITION[0] + distance * math.cos(perturbed_mouse_angle),
                self.config.START_POSITION[1] + distance * math.sin(perturbed_mouse_angle)
            ]

        return pygame.mouse.get_pos()  # No perturbation

    # ========================== MAIN GAME LOOP ==========================
    def run(self):
        """Main game loop - identical in functionality to the original version."""
        running = True

        while running:
            self.screen.fill(self.config.BLACK)

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




            # # ========================== EXPERIMENT DESIGN ==========================
            self.update_experiment_design()

            # Stop experiment when reaching attempt limit
            if self.attempts >= self.config.ATTEMPTS_LIMIT:
                running = False

            # Hide the mouse cursor
            pygame.mouse.set_visible(False)

            # ========================== TRACK MOUSE MOVEMENT ==========================
            mouse_pos = pygame.mouse.get_pos()
            deltax, deltay = mouse_pos[0] - self.config.START_POSITION[0], mouse_pos[1] - self.config.START_POSITION[1]
            distance, mouse_angle = math.hypot(deltax, deltay), math.atan2(deltay, deltax)

            # ========================== CALCULATE PERTURBATION ==========================
            if self.perturbation_mode:
                if self.perturbation_type == 'sudden':
                    perturbed_mouse_angle = mouse_angle + self.config.PERTURBATION_ANGLE
                elif self.perturbation_type == 'gradual':
                    perturbed_mouse_angle = mouse_angle - self.gradual_step * (self.config.PERTURBATION_ANGLE / 10)

                    if self.gradual_attempts % 3 == 0 and self.prev_gradual_attempts != self.gradual_attempts:
                        self.gradual_step += 1

                    if self.gradual_step >= 10:
                        self.gradual_attempts = 0  # Reset after 10 steps for cyclic perturbation


                circle_pos = [
                    self.config.START_POSITION[0] + distance * math.cos(perturbed_mouse_angle + self.config.noise),
                    self.config.START_POSITION[1] + distance * math.sin(perturbed_mouse_angle + self.config.noise)
                ]
            else:
                circle_pos = pygame.mouse.get_pos()

            # ========================== CHECK TARGET HIT OR MISS ==========================
            hit = False

            if self.check_target_reached(circle_pos):
                self.score += 1
                self.attempts += 1
                hit = True

                error_angle = self.compute_error_angle(self.config.START_POSITION, self.new_target, circle_pos)
                if error_angle is not None:
                    self.participant.log_trial(self.attempts, error_angle, hit)

                self.new_target = None  # Reset target
                self.start_time = 0  # Reset timer
                if self.perturbation_type == 'gradual' and self.perturbation_mode:
                    self.gradual_attempts += 1

            # miss if player leaves the target_radius + 1% tolerance
            elif self.new_target and math.hypot(circle_pos[0] - self.config.START_POSITION[0],
                                                circle_pos[1] - self.config.START_POSITION[1]) > self.config.TARGET_RADIUS * 1.01:
                self.attempts += 1
                hit = False

                error_angle = self.compute_error_angle(self.config.START_POSITION, self.new_target, circle_pos)
                if error_angle is not None:
                    self.participant.log_trial(self.attempts, error_angle, hit)

                self.new_target = None  # Reset target
                self.start_time = 0  # Reset timer

                if self.perturbation_type == 'gradual' and self.perturbation_mode:
                    self.gradual_attempts += 1

            # ========================== CHECK IF NEW TARGET IS NEEDED ==========================
            if not self.new_target and self.at_start_position_and_generate_target(mouse_pos):
                self.new_target = self.generate_target_position()
                self.move_faster = False
                self.start_time = pygame.time.get_ticks()

            # ========================== TIME LIMIT CHECK ==========================
            if self.start_time and (pygame.time.get_ticks() - self.start_time) > self.config.TIME_LIMIT:
                self.move_faster = True
                self.start_time = 0  # Reset timer

            # ========================== DISPLAY 'MOVE FASTER' WARNING ==========================
            if self.move_faster:
                font = pygame.font.Font(None, 36)
                text = font.render('MOVE FASTER!', True, self.config.RED)
                text_rect = text.get_rect(center=self.config.START_POSITION)
                self.screen.blit(text, text_rect)
                # Exclude attempts
                if len(self.error_angles) > 0:
                    self.error_angles.pop()  # Remove the most recent one
                    self.error_angles.append(np.nan)

            # ========================== DRAW GAME ELEMENTS ==========================
            if self.new_target:
                pygame.draw.circle(self.screen, self.config.BLUE, self.new_target, self.config.TARGET_SIZE // 2)

            if self.mask_mode:
                if distance < self.config.MASK_RADIUS:
                    pygame.draw.circle(self.screen, self.config.WHITE, circle_pos, self.config.CIRCLE_SIZE // 2)
            else:
                pygame.draw.circle(self.screen, self.config.WHITE, circle_pos, self.config.CIRCLE_SIZE // 2)

            # Draw start position
            pygame.draw.circle(self.screen, self.config.WHITE, self.config.START_POSITION, 5)


            # ========================== DISPLAY SCORE & ATTEMPTS ==========================
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, self.config.WHITE)
            self.screen.blit(score_text, (10, 10))

            attempts_text = font.render(f"Attempts: {self.attempts}", True, self.config.WHITE)
            self.screen.blit(attempts_text, (10, 30))

            # ========================== DEBUGGING INFO (IF TOGGLED) ==========================
            if self.show_mouse_info:
                mouse_info_text = font.render(f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, self.config.WHITE)
                delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, self.config.WHITE)
                mouse_angle_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, self.config.WHITE)
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Motor Learning Experiment')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--phase', type=int, required=True, help='Phase of the experiment')
    args = parser.parse_args()

    task_config = TaskConfig(args.experiment_name, args.phase)

    participant = Participant(participant_id='EH',
                              age=23,
                              gender="F",
                              handedness="Right",
                              task_config=task_config)
    experiment = MotorLearningExperiment(participant,
                                         task_config=task_config)
    experiment.run()

