import torch

def get_embedding_size(embedding_type: str = 'flattened_output'):
    match embedding_type:
        case "final_hidden_state":
            return 80
        case "final_cell_state":
            return 80
        case "concat_final_hidden_cell_state":
            return 160
        case "mean_pooling_output":
            return 16
        case "max_pooling_output":
            return 16
        case "flattened_output":
            return 640

def get_embedding(encoder, input_sequence, embedding_type):
    output, (h_n, c_n) = encoder(input_sequence)
    
    match embedding_type:
        case "final_hidden_state":
            embedding = h_n.reshape(-1)
        case "final_cell_state":
            embedding = c_n.reshape(-1)
        case "concat_final_hidden_cell_state":
            embedding = torch.cat([h_n, c_n], dim=1).reshape(-1)
        case "mean_pooling_output":
            embedding = torch.mean(output, dim=1).reshape(-1)
        case "max_pooling_output":
            embedding = torch.max(output, dim=1).values.reshape(-1)
        case "flattened_output":
            embedding = output.reshape(-1)
    
    return embedding