
def run_fedavg(global_model, clients, rounds=3, local_epochs=1):
    for r in range(rounds):
        print(f"\n🌍 Round {r+1}")

        client_states = []

        for client in clients:
            state = client.train(global_model, epochs=local_epochs)
            client_states.append(state)

        from src.fl.server import fedavg_aggregate
        global_model = fedavg_aggregate(global_model, client_states)

    return global_model
