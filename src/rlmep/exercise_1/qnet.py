import torch
from typing import Callable


class Qnet(torch.nn.Module):
    def __init__(
        self,
        learning_rate=1e-3,
        hidden_dim=16,
        input_dim=1,
        output_dim=4,
        network_kwargs: dict | None = None,
        state_transform: Callable | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if network_kwargs is None:
            network_kwargs = {}

        if state_transform is None:
            state_transform = lambda x: x # noqa: E731

        self.network = self.build_network(input_dim, hidden_dim, output_dim, **network_kwargs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.state_transform = state_transform


    def build_network(self, input_dim, hidden_dim, output_dim=4, **kwargs):
        network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        return network

    def forward(self, x):
        x = self.state_transform(x)
        q = self.network(x).view(-1, self.output_dim)
        return q

    def train(self, states, actions, targets):
        # Forward pass
        outputs = self.forward(states)
        selected_outputs = outputs[range(outputs.shape[0]), actions]

        # Compute loss
        loss = torch.nn.functional.mse_loss(selected_outputs, targets)

        # Update weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().numpy()