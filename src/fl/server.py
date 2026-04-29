
import torch

def fedavg_aggregate(global_model, client_states):
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_state[key] for client_state in client_states],
            dim=0
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model
