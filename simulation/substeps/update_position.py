from typing import Dict, Any
import torch
import random
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

device = "cpu"


@Registry.register_substep("update_position", "transition")
class UpdatePosition(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get("learnable", False):
                setattr(self, key, value["value"])

    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """Update agents moving along graph edges (vectorized version)."""
        # Get current edges and edge progress
        current_edge = get_var(state, self.input_variables["current_edge"])  # shape: [num_agents, 3]
        edge_progress = get_var(state, self.input_variables["edge_progress"])  # shape: [num_agents, 1]
        graph = get_var(state, "environment/graph")

        current_edge = current_edge.to(device)
        edge_progress = edge_progress.to(device)
        graph = graph.to(device)

        # Ensure current_edge dtype: first two columns int64, third float32
        current_edge_indices = current_edge[:, :2].long()
        current_edge_weights = current_edge[:, 2].float()

        # Update progress along edge
        step_size = 0.01 / current_edge_weights.unsqueeze(1)
        edge_progress = edge_progress + step_size

        # Handle agents finishing their edge
        finished_mask = (edge_progress >= 1.0).squeeze()
        if finished_mask.any():
            next_hop = action["citizens"]["next_hop"]
            nh = next_hop[finished_mask]
            valid_mask = nh >= 0

            if valid_mask.any():
                idx = torch.nonzero(finished_mask).squeeze(1)[valid_mask]

                target_nodes = current_edge_indices[idx, 1]
                nh_valid = nh[valid_mask].long()

                current_edge_indices[idx, 0] = target_nodes
                current_edge_indices[idx, 1] = nh_valid
                current_edge_weights[idx] = graph[target_nodes, nh_valid]
                edge_progress[idx] = 0.0

        current_edge = torch.cat([current_edge_indices.to(current_edge.dtype), current_edge_weights.unsqueeze(1)], dim=1)

        outputs = {
            self.output_variables[0]: current_edge,
            self.output_variables[1]: edge_progress,
        }
        return outputs
