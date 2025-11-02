from typing import Dict, Any
import torch
import random
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var


@Registry.register_substep("update_position", "transition")
class UpdatePosition(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get("learnable", False):
                setattr(self, key, value["value"])

    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """Update agents moving along graph edges."""
        # Get current edges and edge progress
        current_edge = get_var(state, self.input_variables["current_edge"])  # shape: [num_agents, 3]
        edge_progress = get_var(state, self.input_variables["edge_progress"])  # shape: [num_agents, 1]
        graph = get_var(state, "environment/graph")
        device = edge_progress.device

        # Use directions from action as step size along edges
        step_size = 0.01 / current_edge[:, 2].unsqueeze(1)
        edge_progress = edge_progress + step_size

        # Handle agents finishing their edge
        finished_mask = edge_progress >= 1.0
        if finished_mask.any():
            finished_agents = finished_mask.nonzero(as_tuple=False).reshape(-1)
            next_hop = action["citizens"]["next_hop"]
            for idx in finished_agents:
                nh = next_hop[idx].item()  # scalar
                if nh >= 0:
                    start_node, target_node, _ = current_edge[idx]
                    start_node = int(start_node.item())
                    target_node = int(target_node.item())
                    nh = int(nh)
                    current_edge[idx, 0] = target_node
                    current_edge[idx, 1] = nh
                    current_edge[idx, 2] = graph[target_node, nh]
                    edge_progress[idx] = 0

        outputs = {
            self.output_variables[0]: current_edge,
            self.output_variables[1]: edge_progress,
        }
        return outputs
