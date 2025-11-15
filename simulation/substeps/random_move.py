from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

device = "cpu"


@Registry.register_substep("move", "policy")
class RandomMove(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """
        Determine the next node (next_hop) for each agent based on current_edge and graph adjacency.
        """
        current_edge = get_var(state, self.input_variables["current_edge"])
        graph = get_var(state, "environment/graph")
        edge_progress = get_var(state, self.input_variables["edge_progress"])

        current_edge = current_edge.to(device)
        graph = graph.to(device)
        edge_progress = edge_progress.to(device)

        num_agents = current_edge.shape[0]

        next_hop = torch.full((num_agents,), -1, dtype=torch.long, device=device)  # shape [num_agents]

        step_size = 0.1 / current_edge[:, 2]

        # Only compute next_hop if edge_progress + step_size >= 1
        move_mask = (edge_progress.squeeze(1) + step_size) >= 1.0
        agents_to_move = move_mask.nonzero(as_tuple=False).view(-1)

        if agents_to_move.numel() > 0:
            target_nodes = current_edge[agents_to_move, 1].long()
            neighbor_mask = graph[target_nodes] > 0
            probs = neighbor_mask.float() / neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1)
            sampled_neighbors = torch.multinomial(probs, 1).squeeze(1)
            no_neighbors = neighbor_mask.sum(dim=1) == 0
            sampled_neighbors[no_neighbors] = target_nodes[no_neighbors]
            next_hop[agents_to_move] = sampled_neighbors

        outputs = {self.output_variables[0]: next_hop}
        return outputs
