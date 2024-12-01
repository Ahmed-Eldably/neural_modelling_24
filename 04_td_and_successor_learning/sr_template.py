import numpy as np
import matplotlib.pyplot as plt

# define maze
maze = np.zeros((9, 13))

# place walls
maze[2, 6:10] = 1
maze[-3, 6:10] = 1
maze[2:-3, 6] = 1

# define start
start = (5, 7)

def plot_maze(maze):
    plt.imshow(maze, cmap='binary')

    # draw thin grid
    for i in range(maze.shape[0]):
        plt.plot([-0.5, maze.shape[1]-0.5], [i-0.5, i-0.5], c='gray', lw=0.5)
    for i in range(maze.shape[1]):
        plt.plot([i-0.5, i-0.5], [-0.5, maze.shape[0]-0.5], c='gray', lw=0.5)

    plt.xticks([])
    plt.yticks([])

plot_maze(maze)
plt.scatter(start[1], start[0], marker='*', color='blue', s=100)
plt.tight_layout()
# plt.savefig('maze.png')
plt.show()


####################################
############## Part 1 ##############
####################################


def random_walk(maze, start, n_steps):
    # perform a single random walk in the given maze, starting from start, performing n_steps random moves
    # moves into the wall and out of the maze boundary are not possible

    # initialize list to store positions
    positions = [start]

    # Current Position
    current_position = np.array(start)

    # Define possible moves
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    # perform random steps...
    for _ in range(n_steps):
        while True:
            # Pick a random move
            move = moves[np.random.choice(len(moves))]

            # Compute the new position
            next_position = current_position + np.array(move)

            # Check if the move is valid
            if (
                    0 <= next_position[0] < maze.shape[0] and  # Vertical bounds
                    0 <= next_position[1] < maze.shape[1] and  # Horizontal bounds
                    maze[next_position[0], next_position[1]] == 0  # not a wall
            ):
                current_position = next_position  # Update position
                break

        # Append the valid position
        positions.append(tuple(current_position))

    # return a list of positions
    return positions

def plot_path(maze, path):
    # plot a maze and a path in it
    plot_maze(maze)
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], c='red', lw=3)
    plt.scatter(path[0, 1], path[0, 0], marker='*', color='blue', s=100)
    plt.scatter(path[-1, 1], path[-1, 0], marker='*', color='green', s=100)
    plt.show()

# plot a random path
path = random_walk(maze, start, 40)
plot_path(maze, path)


####################################
############## Part 2 ##############
####################################


def learn_from_traj(succ_repr, trajectory, maze_shape, gamma=0.98, alpha=0.02):
    # Write a function to update a given successor representation (for the state at which the trajectory starts) using an example trajectory
    # using discount factor gamma and learning rate alpha
    # Grid shape

    # Convert trajectory grid positions to linear indices
    linear_trajectory = [pos[0] * maze_shape[1] + pos[1] for pos in trajectory]

    # Get the starting state index
    start_state = linear_trajectory[0]

    # Initialize discounted trajectory vector
    discounted_trajectory = np.zeros_like(succ_repr[start_state])

    # Populate the discounted trajectory
    for t, future_state in enumerate(linear_trajectory):
        discounted_trajectory[future_state] += gamma ** t

    # Update the successor representation for the starting state
    succ_repr[start_state] = (1 - alpha) * succ_repr[start_state] + alpha * discounted_trajectory

    # return the updated successor representation
    return succ_repr

# initialize successor representation
n_states = maze.shape[0] * maze.shape[1]  # Total number of states (9 * 13 = 117)
succ_repr = np.zeros((n_states, n_states)) # Initialization as 117x117

# sample a whole bunch of trajectories (reduce this number if this code takes too long, but it shouldn't take longer than a minute with reasonable code)
for i in range(5001):
    # sample a path (we use 340 steps here to sample states until the discounting becomes very small)
    path = random_walk(maze, start, 340)
    # update the successor representation
    succ_repr = learn_from_traj(succ_repr, path, maze.shape, alpha=0.02)  # choose a small learning rate

    # occasionally plot it
    if i in [0, 10, 100, 1000, 5000]:
        start_state_index = start[0] * maze.shape[1] + start[1]
        reshaped_sr = succ_repr[start_state_index].reshape(maze.shape)

        plot_maze(maze)
        plt.imshow(reshaped_sr, cmap='hot')
        plt.title(f"SR Visualization After {i} Updates")
        plt.colorbar(label="SR Value")
        plt.show()


####################################
############## Part 3 ##############
####################################


def compute_transition_matrix(maze):
    # for a given maze, compute the transition matrix from any state to any other state under a random walk policy
    # (you will need to think of a good way to map any 2D grid coordinates onto a single number for this)

    # create a matrix over all state pairs
    grid_shape = maze.shape
    n_states = grid_shape[0] * grid_shape[1] # Total number of states ( 9x13 grid)
    transition_matrix = np.zeros((n_states, n_states)) # Initialize transition matrix

    # Define possible moves:
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, Down, Left, Right

    # iterate over all states, filling in the transition probabilities to all other states on the next step (only one step into the future)
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            # Linear index for the current state
            current_state = row * grid_shape[1] + col

            # Skip walls
            if maze[row, col] == 1:
                continue  # Leave the row as zeros (no transitions possible)

            # Compute valid transitions
            valid_transitions = []
            for move in moves:
                next_row, next_col = row + move[0], col + move[1]
                if (
                    0 <= next_row < grid_shape[0] and  # Within row bounds
                    0 <= next_col < grid_shape[1] and  # Within column bounds
                    maze[next_row, next_col] == 0  # Not a wall
                ):
                    next_state = next_row * grid_shape[1] + next_col
                    valid_transitions.append(next_state)

                    # Assign equal probabilities to all valid transitions
                    for next_state in valid_transitions:
                        transition_matrix[current_state, next_state] = 1 / len(valid_transitions)


    
    # normalize transitions if necessary
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for wall states
    transition_matrix = transition_matrix / row_sums

    # remove NaNs if necessary
    transition_matrix = np.nan_to_num(transition_matrix)

    return transition_matrix


####################################
############## Part 4 ##############
####################################


def compute_sr(transitions, i, j, maze_shape, gamma=0.98):
    # given a transition matrix and a specific state (i, j), compute the successor representation of that state with discount factor gamma
    n_states = transitions.shape[0]

    # Convert (i, j) coordinates to linear index
    start_state = i * maze_shape[1] + j

    # initialize things (better to represent the current discounted occupancy as a vector here)
    current_discounted_occupancy = np.zeros(n_states)
    current_discounted_occupancy[start_state] = 1  # Start state occupancy is 1

    # Initialize total SR vector
    total = np.zeros(n_states)

    # iterate for a number of steps
    for step in range(340):  # Steps to ensure discounting becomes negligible
        # Update total SR
        total += current_discounted_occupancy

        # Propagate to the next step
        current_discounted_occupancy = gamma * (current_discounted_occupancy @ transitions)

    # return the successor representation, maybe reshape your vector into the maze shape now
    return total.reshape(maze_shape)

transitions = compute_transition_matrix(maze)

# compute state representation for start state
i, j = start
sr = compute_sr(transitions, i, j, maze.shape, 0.98)

# plot state representation
plot_maze(maze)
plt.imshow(sr, cmap='hot')
plt.title("Iterative SR")
plt.colorbar(label="SR Value")
plt.show()


############################################
############## Part 5 (Bonus) ##############
############################################

# You're on your own now

def compute_sr_analytical(transitions, gamma=0.98):
    """
       Compute the successor representation (SR) analytically for all states.

       Parameters:
       - transitions: Transition matrix (117 x 117).
       - gamma: Discount factor.

       Returns:
       - sr_matrix: Successor representation matrix (117 x 117).
   """
    # Identity matrix
    identity = np.eye(transitions.shape[0])

    # Compute the SR matrix using (I - gamma * T)^-1
    sr_matrix = np.linalg.inv(identity - gamma * transitions)

    return sr_matrix

# Compute the transition matrix
transitions = compute_transition_matrix(maze)

# Compute the analytical SR for the entire grid
analytical_sr = compute_sr_analytical(transitions, gamma=0.98)

# Extract and reshape the SR for the starting state
start_state_index = start[0] * maze.shape[1] + start[1]
start_sr_analytical = analytical_sr[start_state_index].reshape(maze.shape)

# Visualize the analytical SR for the starting state
plot_maze(maze)
plt.imshow(start_sr_analytical, cmap='hot', alpha=0.7)
plt.title("Analytical SR for Starting State")
plt.colorbar(label='SR Value')
plt.show()