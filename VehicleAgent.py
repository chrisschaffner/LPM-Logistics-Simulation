import agenttorch as at
import torch

class TruckAgent(at.Agent):
    def __init__(self, id, position, speed, route=None):
        super().__init__(id=id)
        self.position = torch.tensor(position, dtype=torch.float32)  # (x, y)
        self.speed = speed  # m/s or km/h
        self.route = route  # list of coordinates or node IDs
        self.route_progress = 0.0
        self.cargo = None
        self.state = "idle"

    def step(self, env):
        # Move along the route in continuous space
        if self.route is not None and self.route_progress < len(self.route) - 1:
            curr = torch.tensor(self.route[int(self.route_progress)], dtype=torch.float32)
            next_point = torch.tensor(self.route[int(self.route_progress) + 1], dtype=torch.float32)
            
            # Direction vector
            direction = next_point - curr
            distance = torch.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            
            # Move
            new_position = self.position + direction * self.speed * env.dt
            
            # Check if we’ve reached next point
            if torch.linalg.norm(new_position - next_point) < 1e-2:
                self.route_progress += 1
            
            self.position = new_position