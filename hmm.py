import torch

def hmm_decode(log_pred, transition_penalty=100.0):
    batch_size, seq_len, n_classes = log_pred.shape
    alpha = torch.zeros((batch_size, seq_len, n_classes))
    prev = torch.zeros((batch_size, seq_len, n_classes), dtype=torch.long)
    alpha[:, 0, :] = log_pred[:, 0, :]
    for i in range(1, seq_len):
        # Use matrix operations and avoid for loops
        self_transition = alpha[:, i - 1, :] # [batch_size, n_classes]
        (prev_best, prev_best_id) = torch.max(alpha[:, i - 1, :], dim=-1) # [batch_size]
        non_self_transition = prev_best.unsqueeze(-1).repeat(1, n_classes) - transition_penalty # [batch_size, n_classes]
        prev[:, i, :] = torch.arange(n_classes).unsqueeze(0)
        non_self_transition_classes = non_self_transition > self_transition
        prev[:, i, :][non_self_transition_classes] = prev_best_id.unsqueeze(-1)
        alpha[:, i, :] = log_pred[:, i, :] + torch.max(torch.stack([self_transition, non_self_transition], dim=-1), dim=-1)[0]
    # Backtrack to find the best path
    best_path = torch.zeros((batch_size, seq_len), dtype=torch.long)
    best_path[:, -1] = torch.argmax(alpha[:, -1, :], dim=-1)
    for i in range(seq_len - 2, -1, -1):
        best_path[:, i] = prev[:, i + 1, best_path[:, i + 1]]
    return best_path
