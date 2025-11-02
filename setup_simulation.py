from pathlib import Path
import random
from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder,
)

import math

def create_random_2d_graph(num_nodes, connection_prob=0.25, min_dist=0.2, max_dist=0.8):
    """Generate random 2D positions and adjacency matrix with weights as Euclidean distances."""
    # Random 2D positions in unit square
    positions = [
        [random.uniform(0, 1), random.uniform(0, 1)]
        for _ in range(num_nodes)
    ]
    adjacency_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < connection_prob:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dist = math.hypot(x2 - x1, y2 - y1)
                if min_dist <= dist <= max_dist:
                    adjacency_matrix[i][j] = dist
                    adjacency_matrix[j][i] = dist
    return adjacency_matrix, positions

def setup_movement_simulation():
    """Setup a complete movement simulation structure."""
    
    # Create config builder
    config = ConfigBuilder()

    # Set simulation metadata - these are all required arguments
    metadata = {
        "num_agents": 500,
        "num_episodes": 5,
        "num_steps_per_episode": 200,
        "num_substeps_per_step": 1,
        "num_nodes": 45,
        "device": "cpu",
        "calibration": False,
    }
    config.set_metadata(metadata)

    adjacency_matrix, node_positions = create_random_2d_graph(metadata["num_nodes"])

    # Build state
    state_builder = StateBuilder()

    # Add agent
    agent_builder = AgentBuilder("citizens", metadata["num_agents"])

    # Agent properties for movement along edges
    # edges: list of [start_node, end_node, edge_length]
    edges = [
        [i, j, adjacency_matrix[i][j]]
        for i in range(metadata["num_nodes"])
        for j in range(metadata["num_nodes"])
        if i != j and adjacency_matrix[i][j] > 0
    ]
    # Assign a random edge to each agent (with edge length)
    initial_edges = [random.choice(edges) for _ in range(metadata["num_agents"])]
    current_edge = (
        PropertyBuilder("current_edge")
        .set_dtype("float")
        .set_shape([metadata["num_agents"], 3])
        .set_value(initial_edges)
    )
    edge_progress = (
        PropertyBuilder("edge_progress")
        .set_dtype("float")
        .set_shape([metadata["num_agents"], 1])
        .set_value([0.0] * metadata["num_agents"])
    )
    agent_builder.add_property(current_edge)
    agent_builder.add_property(edge_progress)
    state_builder.add_agent("citizens", agent_builder)

    # Add environment variables: adjacency matrix and node positions
    env_builder = EnvironmentBuilder()
    graph_property = (
        PropertyBuilder("graph")
        .set_dtype("float")
        .set_shape([metadata["num_nodes"], metadata["num_nodes"]])
        .set_value(adjacency_matrix)
    )
    positions_property = (
        PropertyBuilder("node_positions")
        .set_dtype("float")
        .set_shape([metadata["num_nodes"], 2])
        .set_value(node_positions)
    )
    env_builder.add_variable(graph_property)
    env_builder.add_variable(positions_property)
    state_builder.set_environment(env_builder)

    # Set state in config
    config.set_state(state_builder.to_dict())

    # Create movement substep
    movement = SubstepBuilder("Movement", "Agent movement simulation")
    movement.add_active_agent("citizens")

    # Set observation structure
    movement.config["observation"] = {"citizens": None}

    # Add movement policy
    policy = PolicyBuilder()
    step_size = PropertyBuilder.create_argument(
        name="Step size parameter", value=1.0, learnable=True
    ).config

    policy.add_policy(
        "move",
        "RandomMove",  # custom policy to move along edges
        {
            "current_edge": "agents/citizens/current_edge",
            "edge_progress": "agents/citizens/edge_progress",
            "graph": "environment/graph",
        },
        ["next_hop"],
        {"step_size": step_size},
    )
    movement.set_policy("citizens", policy)

    # Add movement transition
    transition = TransitionBuilder()
    transition.add_transition(
        "update_position",
        "UpdatePosition",
        {
            "current_edge": "agents/citizens/current_edge",
            "edge_progress": "agents/citizens/edge_progress",
        },
        ["current_edge", "edge_progress"],
        {},
    )
    movement.set_transition(transition)

    # Add substep to config
    config.add_substep("0", movement)

    # Save the config
    examples_dir = Path(__file__).parent
    model_dir = examples_dir / "simulation"
    config_path = model_dir / "yamls" / "config.yaml"
    config.save_yaml(str(config_path))
    print(f"\nGenerated config file: {config_path}")

    return config

if __name__ == "__main__":
    setup_movement_simulation()