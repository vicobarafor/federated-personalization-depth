# Methodology

The study isolates personalization depth after federated pretraining.

The protocol is:

1. Train a shared global model using FedAvg.
2. Broadcast the shared model to each client.
3. Personalize each client using one of three adaptation depths.
4. Compare fixed policies against client-wise oracle routing.

The candidate personalization depths are head-only fine-tuning, partial fine-tuning, and full fine-tuning.
