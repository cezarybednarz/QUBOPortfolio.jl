import numpy as np
import simulated_bifurcation as sb
import json
import sys
import torch
import time

def solve_qubo(edges, n, hyperparameters):
    """
    Solves a QUBO problem using the simulated bifurcation algorithm.

    The input payload should be a JSON object with the following structure:
      { "n": n, "edges": [[i, j, value], ...], "hyperparameters": {...} }

    Indices i, j are 0-based. The function rebuilds the full Q matrix from
    the adjacency list (symmetrizing entries where necessary) and then
    runs the simulated bifurcation minimizer.

    Args:
        payload_json (str): JSON string representing the adjacency-list payload.

    Returns:
        str: JSON string with keys "result_bits" and "energy".
    """
    
    Q = np.zeros((n, n), dtype=np.float32)
    for e in edges:
        i, j, val = e
        i = int(i)
        j = int(j)
        v = float(val)
        Q[i, j] += v
    Q = (Q + Q.T) / 2.0
    Q_torch = torch.from_numpy(Q)

    # Set hyperparameters for SB
    if ("time_step" in hyperparameters):
        sb.set_env(time_step=hyperparameters["time_step"])
    if ("pressure_slope" in hyperparameters):
        sb.set_env(pressure_slope=hyperparameters["pressure_slope"])
    if ("heat_coefficient" in hyperparameters):
        sb.set_env(heat_coefficient=hyperparameters["heat_coefficient"])

    # run on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run the simulated bifurcation minimizer
    all_vectors, all_energies = sb.minimize(
        Q_torch,
        device=device,
        domain="binary", # QUBO domain
        verbose=False,  # Hide progress bars
        # those are included as hyperparameters as well:
        mode=hyperparameters.get("mode", "discrete"),  # Mode of operation ("ballistic" or "discrete")
        heated=hyperparameters.get("heated", False),  # Whether to use heated SB
        max_steps=hyperparameters.get("max_steps", 1000),  # Maximum number of steps
        agents=hyperparameters.get("agents", 1024)  # Number of parallel agents
    )

    all_vectors = all_vectors.unsqueeze(0)
    all_energies = all_energies.unsqueeze(0)

    best_energy_tensor, best_solution_index = torch.min(all_energies, dim=0)
    best_vector_tensor = all_vectors[best_solution_index.item()]

    energy = best_energy_tensor.item()
    vector_np = best_vector_tensor.cpu().numpy().astype(int)

    result = {
        "result_bits": vector_np.tolist(),
        "energy": energy
    }

    return json.dumps(result)


if __name__ == "__main__":
    payload_json = sys.stdin.read()
    data = json.loads(payload_json)
    edges = data['edges']
    n = data['n']
    hyperparameters = data.get('hyperparameters', {})

    result_json = solve_qubo(edges, n, hyperparameters)

    print(result_json)