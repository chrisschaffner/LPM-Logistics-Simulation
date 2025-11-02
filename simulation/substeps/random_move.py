from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var


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

        device = current_edge.device
        num_agents = current_edge.shape[0]

        next_hop = torch.full((num_agents,), -1, dtype=torch.long, device=device)  # shape [num_agents]


        step_size = 0.1 / current_edge[:, 2]

        # Only compute next_hop if edge_progress + step_size >= 1
        move_mask = (edge_progress.squeeze(1) + step_size) >= 1.0
        agents_to_move = move_mask.nonzero(as_tuple=False).view(-1)

        for idx in agents_to_move:
            start_node, target_node, edge_length = current_edge[idx]
            start_node = int(start_node.item())
            target_node = int(target_node.item())
            neighbors = graph[target_node].nonzero(as_tuple=False).flatten()
            if len(neighbors) == 0:
                next_hop[idx] = target_node
            else:
                next_hop[idx] = neighbors[torch.randint(0, len(neighbors), (1,), device=device)].item()
        outputs = {self.output_variables[0]: next_hop}
        return outputs
