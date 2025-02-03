import datetime
import os
import time
import json

import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Pygame
pygame.init()

# Screen settings
# Full screen mode
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
pygame.display.set_caption("Beer Pint Game")

# Constants
TABLE_WIDTH = SCREEN_WIDTH - 100
TABLE_HEIGHT = int(SCREEN_HEIGHT * 0.9)
FREE_ZONE_RADIUS = 110
START_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - FREE_ZONE_RADIUS - 10)

# Colors
WHITE = (255, 255, 255)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)
LIGHT_RED = (255, 182, 193)
DARK_BROWN = (120, 66, 40)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
GREEN_LAMP = (0, 255, 0)
RED_LAMP = (255, 0, 0)

# Game settings
BASE_FRICTION = 0.99
ZONE_WIDTH = int(TABLE_WIDTH * 0.95)
ZONE_HEIGHT = 150

# Rectangles and triangles
SCORING_RECT = pygame.Rect(
    (SCREEN_WIDTH - ZONE_WIDTH) // 2,
    int(TABLE_HEIGHT * 0.2),
    ZONE_WIDTH,
    ZONE_HEIGHT,
)

TABLE_RECT = pygame.Rect((SCREEN_WIDTH - TABLE_WIDTH) // 2, (SCREEN_HEIGHT - TABLE_HEIGHT) // 2, TABLE_WIDTH, TABLE_HEIGHT)

GREEN_TRIANGLE = [
    SCORING_RECT.topleft,
    SCORING_RECT.topright,
    SCORING_RECT.bottomleft
]
RED_TRIANGLE = [
    SCORING_RECT.bottomright,
    SCORING_RECT.bottomleft,
    SCORING_RECT.topright
]

# Game variables
pint_pos = list(START_POS)
pint_velocity = [0, 0]
pint_radius = 15
friction = BASE_FRICTION
launched = False
stopped = False
waiting_for_mouse = True
perturbation_active = False
score = 0
feedback_mode = False
feedback_type = None
trajectory = []
perturbation_force=0
force_increment=0.2
end_pos = [0, 0]
current_block = 1
show_info=1
trial_positions = []
last_trajectory=[]
noise_mean = None
noise_std = None
noise_x = None
noise_y = None
pint_color = YELLOW


# Font setup
font = pygame.font.SysFont(None, 36)

# BUILD FIELD
def draw_playfield(mask_pint=False):
    """Draw the game playfield."""
    screen.fill(WHITE)

    # Draw the table
    pygame.draw.rect(screen, DARK_BROWN, TABLE_RECT)

    # Draw free movement zone
    pygame.draw.circle(screen, LIGHT_BLUE, START_POS, FREE_ZONE_RADIUS)
    pygame.draw.circle(screen, BLACK, START_POS, FREE_ZONE_RADIUS, 3)

    # Draw scoring areas with precomputed gradients
    screen.blit(green_gradient, SCORING_RECT.topleft)
    screen.blit(red_gradient, SCORING_RECT.topleft)

    # Optionally mask the beer pint
    if not mask_pint:
        pygame.draw.circle(screen, pint_color, (int(pint_pos[0]), int(pint_pos[1])), pint_radius)
        pygame.draw.circle(screen, WHITE, (int(pint_pos[0]), int(pint_pos[1])), pint_radius + 2, 2)

def apply_motor_noise(mean=None, std_dev=None):
    """Apply motor noise with specified mean and standard deviation."""
    global pint_velocity, noise_x, noise_y

    if mean is not None:
        # Generate random noise from a Gaussian distribution
        noise_x = np.random.normal(mean, std_dev)
        noise_y = np.random.normal(mean, std_dev)

        # Apply noise to the movement velocity
        pint_velocity[0] += noise_x
        pint_velocity[1] += noise_y



# Precompute gradient surfaces
def create_gradient_surface(points, start_color, end_color, reference_point):
    """Generate a gradient surface for a triangular region."""
    max_distance = max(math.dist(reference_point, p) for p in points)
    surface = pygame.Surface((SCORING_RECT.width, SCORING_RECT.height), pygame.SRCALPHA)

    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            world_x = SCORING_RECT.left + x
            world_y = SCORING_RECT.top + y
            if point_in_polygon((world_x, world_y), points):
                distance = math.dist((world_x, world_y), reference_point)
                factor = min(distance / max_distance, 1.0)
                color = interpolate_color(start_color, end_color, factor)
                surface.set_at((x, y), color + (255,))  # Add alpha
    return surface

def interpolate_color(start_color, end_color, factor):
    """Interpolate between two colors."""
    return tuple(int(start + (end - start) * factor) for start, end in zip(start_color, end_color))

# PINT_MOVEMENTS
def handle_mouse_input():
    """Handle mouse interactions with the pint."""
    global pint_pos, pint_velocity, launched, waiting_for_mouse
    mouse_pos = pygame.mouse.get_pos()
    distance = math.dist(mouse_pos, START_POS)
    if waiting_for_mouse:
        if distance <= pint_radius:  # Mouse touching the pint
            waiting_for_mouse = False
    elif distance <= FREE_ZONE_RADIUS:
        pint_pos[0], pint_pos[1] = mouse_pos
    else:
        pint_velocity = calculate_velocity(pint_pos, mouse_pos)
        apply_motor_noise(mean=noise_mean,
                          std_dev=noise_std)
        if perturbation_active:
            apply_perturbation()
        launched = True

def calculate_velocity(start_pos, mouse_pos):
    dx = mouse_pos[0] - start_pos[0]
    dy = mouse_pos[1] - start_pos[1]
    speed = math.sqrt(dx**2 + dy**2) / 10
    angle = math.atan2(dy, dx)
    return [speed * math.cos(angle), speed * math.sin(angle)]

def apply_friction():
    global pint_velocity
    pint_velocity[0] *= friction
    pint_velocity[1] *= friction

def update_perturbation():
    """Adjust the perturbation force based on gradual or sudden mode."""
    global perturbation_force, trial_in_block

    if gradual_perturbation and perturbation_active:
        # Increment force every 3 trials (or however you want to adjust the frequency)
        if trial_in_block % 3 == 0 and trial_in_block != 0:
            perturbation_force += force_increment  # Increase perturbation force gradually after each set of 3 trials
            print(f"Gradual perturbation force updated to: {perturbation_force}")
    # Sudden perturbation: No updates needed (force remains constant)
def apply_perturbation():
    """Apply perturbation to the pint's movement."""
    if perturbation_active:
        pint_velocity[0] += perturbation_force  # Add rightward force

# CHECK & SCORE
def check_stopped():
    global stopped, launched, last_trajectory
    if abs(pint_velocity[0]) < 0.1 and abs(pint_velocity[1]) < 0.1 and launched:
        stopped = True
        launched = False
        last_trajectory = trajectory[:]
        if last_trajectory:
            end_pos = last_trajectory[-1]  # Store the final position
            print(f"Last trajectory saved: {last_trajectory}")
            print(f" End position saved: {end_pos}")


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[0]
    for i in range(1, n + 1):
        sx, sy = polygon[i % n]
        if y > min(py, sy):
            if y <= max(py, sy):
                if x <= max(px, sx):
                    if py != sy:
                        xinters = (y - py) * (sx - px) / (sy - py) + px
                    if px == sx or x <= xinters:
                        inside = not inside
        px, py = sx, sy
    return inside

def calculate_score():
    """Calculate and update the score."""
    global pint_pos, stopped, end_pos, score, trial_counter, trial_positions
    if stopped:  # Only calculate score once per trial
        if point_in_polygon(pint_pos, GREEN_TRIANGLE):
            reference_point = SCORING_RECT.topleft
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in GREEN_TRIANGLE)
            score += calculate_edge_score(distance, max_distance)
        elif point_in_polygon(pint_pos, RED_TRIANGLE):
            reference_point = SCORING_RECT.bottomright
            distance = math.dist(pint_pos, reference_point)
            max_distance = max(math.dist(p, reference_point) for p in RED_TRIANGLE)
            score -= calculate_edge_score(distance, max_distance)
        elif not TABLE_RECT.collidepoint(*pint_pos):
            score -= 50  # Penalty for missing
            display_message("Too far!")

        # Append trial position and current block number
        trial_positions.append((pint_pos[0], pint_pos[1], current_block, noise_x, noise_y, pint_velocity[0], pint_velocity[1], friction))
        reset_pint()
        handle_trial_end()



def calculate_edge_score(distance, max_distance):
    """
    Calculate the score based on distance to the reference point.
    100 points for the closest edge, 10 points for the farthest edge.
    """
    normalized_distance = min(distance / max_distance, 1.0)  # Normalize to [0, 1]
    return int(100 - 90 * normalized_distance)  # Scale between 100 and 10

def display_message(text):
    message = font.render(text, True, BLACK)
    screen.blit(message, (SCREEN_WIDTH // 2 - message.get_width() // 2, SCREEN_HEIGHT // 2 - message.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(1000)

def reset_pint():
    """Reset the pint to the starting position."""
    global pint_pos, end_pos, last_trajectory, pint_velocity, launched, stopped, waiting_for_mouse, trajectory
    pint_pos[:] = START_POS
    pint_velocity[:] = [0, 0]
    launched = False
    stopped = False
    waiting_for_mouse = True
    trajectory.clear()


def draw_point(position, radius=pint_radius, flag=False):
    if not flag:
        pygame.draw.circle(screen, YELLOW, (int(position[0]), int(position[1])), radius)
    pygame.draw.circle(screen, WHITE, (int(position[0]), int(position[1])), radius + 2, 2)


#TASK 1: IMPLEMENT FEEDBACK MODES

def draw_feedback(feedback_type, last_trajectory):
    """Display feedback based on the feedback type."""
    print("Feedback type: {}".format(feedback_type))
    print("Last trajectory: {}".format(last_trajectory))
    if feedback_type == "trajectory":
        for point_pos in last_trajectory:
            draw_point(position=point_pos)
    elif feedback_type == "endpos":
        if len(last_trajectory) > 0:
            last_pos = last_trajectory[-1]
            draw_point(position=last_pos, flag=True)
    elif feedback_type == "endpos_approx":
            last_pos = last_trajectory[-1]
            scaling_factor=5
            draw_point(position=last_pos, radius=pint_radius*scaling_factor, flag=True)
    elif feedback_type == "rl":
        # Draw free movement zone
        if point_in_polygon(pint_pos, GREEN_TRIANGLE):
            pygame.draw.circle(screen, DARK_GREEN, START_POS, FREE_ZONE_RADIUS, 5)
        else:
            pygame.draw.circle(screen, DARK_RED, START_POS, FREE_ZONE_RADIUS, 5)



# Precompute gradient surfaces
green_gradient = create_gradient_surface(GREEN_TRIANGLE, DARK_GREEN, LIGHT_GREEN, SCORING_RECT.topleft)
red_gradient = create_gradient_surface(RED_TRIANGLE, DARK_RED, LIGHT_RED, SCORING_RECT.bottomright)


#Design Experiment
def setup_block(block_number):
    """Set up block parameters."""
    global perturbation_active, feedback_mode, feedback_type, perturbation_force, trial_in_block, gradual_perturbation, noise_mean, noise_std, friction, pint_color

    block = block_structure[block_number - 1]
    feedback_type = block['feedback'] if block['feedback'] else None
    feedback_mode = feedback_type is not None

    perturbation_active = block['perturbation']
    trial_in_block = 0

    # Apply global perturbation mode to set gradual or sudden
    if perturbation_active:
        if not block['sudden']:  # Gradual perturbation
            gradual_perturbation = True
            perturbation_force = block.get('initial_force', 0)  # Use the initial force for gradual perturbation
        else:  # Sudden perturbation
            gradual_perturbation = False
            perturbation_force = block.get('sudden_force', 10.0)  # Use the sudden force for sudden perturbation

    noise_mean  = block['noise_mean'] if block['noise_mean'] else None
    noise_std = block['noise_std'] if block['noise_std'] else None

    if 'friction' in block:
        friction = block['friction']
    else:
        friction = BASE_FRICTION

    if 'pint_color' in block:
        pint_color = block['pint_color']



def handle_trial_end():
    """Handle end-of-trial events."""
    global trial_in_block, current_block, running

    trial_in_block += 1

    # Update perturbation force for gradual perturbation
    if perturbation_active and gradual_perturbation:
        update_perturbation()

    # Transition to the next block if trials in the current block are complete
    if trial_in_block >= block_structure[current_block - 1]['num_trials']:
        current_block += 1
        if current_block > len(block_structure):
            running = False  # End experiment
        else:
            setup_block(current_block)

# TASK1: Define the experiment blocks
block_structure = [
    #Normal visual feedback
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': None, 'noise_std': None, 'pint_color': YELLOW},  # 10 trials without perturbation
    {'feedback': 'endpos', 'perturbation': True, 'sudden': True, 'num_trials': 30, 'initial_force': 0.2, 'sudden_force': 2.0, 'noise_mean': None, 'noise_std': None, 'pint_color': YELLOW},  # 30 trials with gradual perturbation
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': None, 'noise_std': None, 'pint_color': YELLOW},  # 10 trials without perturbation
    # ADD Trajectory feedback
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 0.1, 'noise_std': 1.5, 'friction': BASE_FRICTION - 0.03, 'pint_color': (255, 255, 102)},  # 10 trials without perturbation
    {'feedback': 'endpos', 'perturbation': True, 'sudden': True, 'num_trials': 30, 'initial_force': 0.2, 'sudden_force': 2.0, 'noise_mean': 0.1, 'noise_std': 1.5, 'friction': BASE_FRICTION - 0.03, 'pint_color': (255, 255, 102)},  # 30 trials with gradual perturbation
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 0.1, 'noise_std': 1.5, 'friction': BASE_FRICTION - 0.03, 'pint_color': (255, 255, 102)},  # 10 trials without perturbation
    # ADD End position feedback
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 2, 'noise_std': 3, 'friction': BASE_FRICTION - 0.02, 'pint_color': (255, 255, 179)},  # 10 trials without perturbation
    {'feedback': 'endpos', 'perturbation': True, 'sudden': True, 'num_trials': 30, 'initial_force': 0.2, 'sudden_force': 2.0, 'noise_mean': 2, 'noise_std': 3, 'friction': BASE_FRICTION - 0.02, 'pint_color': (255, 255, 179)},  # 30 trials with gradual perturbation
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 2, 'noise_std': 3, 'friction': BASE_FRICTION - 0.02, 'pint_color': (255, 255, 179)},  # 10 trials without perturbation
    # ADD RL feedback
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 4, 'noise_std': 6, 'friction': BASE_FRICTION - 0.01, 'pint_color': WHITE},  # 10 trials without perturbation
    {'feedback': 'endpos', 'perturbation': True, 'sudden': True, 'num_trials': 30, 'initial_force': 0.2, 'sudden_force': 2.0, 'noise_mean': 4, 'noise_std': 6, 'friction': BASE_FRICTION - 0.01, 'pint_color': WHITE},  # 30 trials with gradual perturbation
    {'feedback': 'endpos', 'perturbation': False, 'gradual': False, 'num_trials': 10, 'noise_mean': 4, 'noise_std': 6, 'friction': BASE_FRICTION - 0.01, 'pint_color': WHITE},  # 10 trials without perturbation
   # # # ADD End Position Approximate
   #  {'feedback': None, 'perturbation': False, 'gradual': False, 'num_trials': 10},  # 10 trials without perturbation
   #  {'feedback': 'endpos_approx', 'perturbation': True, 'gradual': True, 'num_trials': 30, 'initial_force': 0.2, 'sudden_force': 2.0},  # 30 trials with gradual perturbation
   #  {'feedback': None, 'perturbation': False, 'gradual': False, 'num_trials': 10},  # 10 trials without perturbation
]

mask_pint = launched and feedback_mode and feedback_type in ('trajectory', 'rl', 'endpos', 'endpos_approx')

current_block = 1
setup_block(current_block)

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
# Determine if the beer pint should be masked
    mask_pint = launched and feedback_mode and feedback_type in ('trajectory', 'rl', 'endpos', 'endpos_approx')

    # Draw playfield with optional masking
    draw_playfield(mask_pint=mask_pint)

    # Display score (only for feedbacks where score is not relevant)
    if feedback_type not in ('rl', 'endpos', 'trajectory', 'endpos_approx'):
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

    # Handle Keyboard events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_4:
                perturbation_mode = True
            elif event.key == pygame.K_5:
                perturbation_mode = False
            elif event.key == pygame.K_1:
                feedback_type = 'trajectory'
                feedback_mode = True
            elif event.key == pygame.K_2:
                feedback_type = 'endpos'
                feedback_mode = True
            elif event.key == pygame.K_3:
                feedback_type = 'rl'
                feedback_mode = True
            elif event.key == pygame.K_0:
                feedback_mode = False
            elif event.key == pygame.K_i:  # Press 'i' to toggle info display
                show_info = not show_info
            elif event.key == pygame.K_SPACE:  # Start the next experimental block
                current_block += 1
                if current_block > len(block_structure):
                    running = False  # End the experiment
                else:
                    setup_block(current_block)

    if launched:
        apply_friction()
        pint_pos[0] += pint_velocity[0]
        pint_pos[1] += pint_velocity[1]
        trajectory.append((int(pint_pos[0]), int(pint_pos[1])))
        check_stopped()
        calculate_score()
    else:
        handle_mouse_input()


    # Draw feedback if applicable
    draw_feedback(feedback_type=feedback_type, last_trajectory=last_trajectory)

    if stopped:
        reset_pint()

    if show_info:
        fb_info_text = font.render(f"Feedback: {feedback_type}", True, BLACK)
        pt_info_text = font.render(f"Perturbation:{perturbation_active}", True, BLACK)
        pf_info_text = font.render(f"Perturbation_force:{perturbation_force}", True, BLACK)
        tib_text = font.render(f"Trial_in_block: {trial_in_block}", True, BLACK)
        current_block_text = font.render(f"current_block: {current_block}", True, BLACK)
        noise_x_y_text = font.render(f"noise_x, noise_y: {noise_x, noise_y}", True, BLACK)
        noise_mean_std_text = font.render(f"noise_mean, noise_std: {noise_mean, noise_std}", True, BLACK)
        friction_text = font.render(f"friction: {friction}", True, BLACK)
        pint_velocity_text = font.render(f"pint_velocity: {pint_velocity}", True, BLACK)


        screen.blit(fb_info_text, (10, 40))
        screen.blit(pt_info_text, (10, 60))
        screen.blit(pf_info_text, (10, 90))
        screen.blit(tib_text, (10, 120))
        screen.blit(current_block_text, (10, 150))
        screen.blit(noise_x_y_text, (10, 180))
        screen.blit(noise_mean_std_text, (10, 210))
        screen.blit(friction_text, (10, 240))
        screen.blit(pint_velocity_text, (10, 270))


    pygame.display.flip()
    clock.tick(60)

pygame.quit()

#TASK 2: PLOT Hitting patterns for all feedbacks
feedback_blocks = {
    'trajectory': [4, 5, 6],
    'endpos': [7, 8, 9],
    'rl': [10, 11, 12],
    'endpos_approx': [13, 14, 15],
    None: [1, 2, 3]  # Normal feedback type
}
#use trial_positions
# Separate trial positions by feedback type
feedback_trials = {key: [] for key in feedback_blocks.keys()}


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
directory_location = f"participant_data"
directory = os.path.join(os.path.dirname(__file__), f'{directory_location}')

if not os.path.exists(directory):
    os.makedirs(directory)

filename = f'{directory}/participant_AN_{timestamp}'
subject_id = f"AN"

trial_positions_final_list = []
for trial in trial_positions:
    trial_list = list(trial)
    trial_list.append(subject_id)
    trial_positions_final_list.append(trial_list)

# Convert trial_positions to DataFrame and explicitly define column names
df = pd.DataFrame(trial_positions_final_list, columns=['x', 'y', 'feedback_block', 'subject_id', 'noise_x', 'noise_y', 'pint_velocity_x', 'pint_velocity_y', 'friction'])

# Save to CSV ensuring column headers are included
df.to_csv(f"{filename}.csv", index=False, header=True)

# Plot results (hitting patterns on table + end score) grouped by feedback type


for pos in trial_positions:
    x, y, block = pos
    for fb_type, blocks in feedback_blocks.items():
        if block in blocks:
            feedback_trials[fb_type].append((x, y))

# Define colors for each feedback type
feedback_colors = {
    'trajectory': 'blue',
    'endpos': 'red',
    'rl': 'green',
    None: 'black'
}

# Plot hitting patterns
