import random
import numpy as np

def si_epidemic(graph, beta, initial_infected_count=1, max_iterations=None, max_infected_frac=None):
    """
    Simulates an SI epidemic model on a given graph.

    Parameters:
    - graph: NetworkX graph
    - beta: Probability of infection per contact
    - initial_infected_count: Number of initually infected nodes
    - max_interations: Maximum number of iterations (None to ignore)
    - max_infected_frac: Maximum fraction of infected nodes before stopping (None to ignore)

    Return:
    - A list containing the infected nodes at the end of the simulation.
    """
    # Ensure that at least one stop condition (max_terations or max_infected_frac) is provided
    if max_iterations is None and max_infected_frac is None:
        raise ValueError("At least onde of max_iterations or max_infected_frac must be provided.")
    
    initial_infected = random.sample(graph.nodes(), initial_infected_count) # Randomly select the initial infected nodes from the graph
    infected = set(initial_infected)
    iteration = 0

    # If a maximum fraction of infected nodes is provided, calculate the maximum number of infected nodes
    max_infected = None if max_infected_frac is None else int(max_infected_frac * len(graph))

    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        if max_infected is not None and len(infected) >= max_infected:
            break

        new_infected = set() # Store new infections in this time step
        infected_list = list(infected)
        random.shuffle(infected_list) # Shuffle list of infected nodes to avoid bias

        for node in infected:
            # Iterate over neighbors of the infected node that are not already infected
            for neighbor in graph.neighbors(node) - infected:
                # With probability 'beta', attempt to infect the neighbor
                if random.random() < beta:
                    new_infected.add(neighbor) # Add the neighbor to the new infected set

        infected.update(new_infected) # Update the infected set with the newly infected nodes
        iteration += 1

    return list(infected)


def observable_infected(infected_nodes, observation_probability):
    """
    Returns a list of observable infected nodes based on a given observation probability.
    
    Parameters:
    - infected_nodes: List of infected nodes.
    - observation_probability: Probability (between 0 and 1) of observing each infected node.
    
    Returns:
    - A list of observable infected nodes.
    """
    infected_nodes_array = np.array(infected_nodes)
    
    # Generate random values between 0 and 1 for each infected node
    random_values = np.random.random(len(infected_nodes_array))
    
    # Select nodes where the random value is less than the observation probability
    observable_nodes = infected_nodes_array[random_values < observation_probability]
    
    return list(observable_nodes)
