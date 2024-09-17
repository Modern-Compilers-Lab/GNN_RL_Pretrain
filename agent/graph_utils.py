import numpy as np 
import re
import torch

def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix

def iterators_to_vectors(annotations):
    it_dict = {}
    iter_vector_size = 655
    size_of_comp_vector = 646
    for it in annotations["iterators"]:
        single_iter_vector = -np.ones(iter_vector_size)
        single_iter_vector[0] = 0
        single_iter_vector[-9:] = 0
        # lower value
        single_iter_vector[size_of_comp_vector + 1] = annotations["iterators"][it][
            "lower_bound"
        ]
        # upper value
        single_iter_vector[size_of_comp_vector + 2] = annotations["iterators"][it][
            "upper_bound"
        ]
        it_dict[it] = single_iter_vector

    return it_dict

def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix

def encode_data_type(data_type):
    if data_type == "int32":
        return [1, 0, 0]
    elif data_type == "float32":
        return [0, 1, 0]
    elif data_type == "float64":
        return [0, 0, 1]

def get_sequence(comp_dict, max_depth, max_accesses):
    sequence = []

    padded_write_matrix = pad_access_matrix(
        isl_to_write_matrix(comp_dict["write_access_relation"]), max_depth
    )
    write_access_repr = [
        -1,
        comp_dict["write_buffer_id"] + 1
    ] + padded_write_matrix.flatten().tolist()
    # print("write access repr: ", len(write_access_repr))

    sequence.append(write_access_repr)

    # Pad the read access matrix and add it to the representation 
    # read_accesses_repr = []
    for read_access_dict in comp_dict["accesses"]:
        read_access_matrix = pad_access_matrix(
            read_access_dict["access_matrix"], max_depth
        )
        read_access_repr = (
            [+read_access_dict["access_is_reduction"]]
            + [read_access_dict["buffer_id"] + 1]
            + read_access_matrix.flatten().tolist()
        )
        sequence.append(read_access_repr)
        # print("read access repr: ", len(read_access_repr))
    access_repr_len = (max_depth + 1) * (max_depth + 2) + 1 + 1

    for i in range(max_accesses - len(comp_dict["accesses"])):
        sequence.append([0]*access_repr_len)
    # print("sequence len: ", len(sequence), len(sequence[0]))    
    return sequence

def get_embedding(encoder, input_sequence):
    output, (h_n, c_n) = encoder(input_sequence)
    
    # # Option 1: Final hidden state
    # final_hidden = h_n.transpose(0, 1).contiguous().view(input_sequence.size(0), -1)
    
    # # Option 2a: Last time step of output
    # last_output = output[:, -1, :]
    
    # # Option 2b: Mean pooling of output
    # mean_output = torch.mean(output, dim=1)
    
    # Option 3: Flattened output sequence
    flattened_output = output.reshape(-1)
    return flattened_output

def comps_to_vectors(annotations, encoder):
    comp_vector_size = 655
    max_depth = 5
    max_accesses = 15
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dict_comp = {}
    for comp in annotations["computations"]:
        single_comp_vector = -np.ones(comp_vector_size)
        # This means that this vector has data related to a computation and not an iterator
        single_comp_vector[0] = 1
        comp_dict = annotations["computations"][comp]
        # This field represents the absolute order of execution of computations
        single_comp_vector[1] = comp_dict["absolute_order"]
        # a vector of one-hot encoding of possible 3 data-types
        single_comp_vector[2:5] = encode_data_type(comp_dict["data_type"])
        single_comp_vector[5] = +comp_dict["comp_is_reduction"]
        access_matrices = get_sequence(comp_dict, max_depth, max_accesses)
        access_matrices = torch.tensor(access_matrices, dtype=torch.float32).to(device)
        access_embedding = get_embedding(encoder, access_matrices)
        single_comp_vector[6:6+640] = access_embedding.cpu().detach().numpy()
        dict_comp[comp] = single_comp_vector
    return dict_comp

def build_graph(annotations, encoder):
    it_vector_dict = iterators_to_vectors(annotations)
    comp_vector_dict = comps_to_vectors(annotations, encoder)
    it_index = {}
    comp_index = {}
    num_iterators = len(annotations["iterators"])
    for i, it in enumerate(it_vector_dict):
        it_index[it] = i
    for i, comp in enumerate(comp_vector_dict):
        comp_index[comp] = i

    edge_index = []
    node_feats = None

    for it in annotations["iterators"]:
        for child_it in annotations["iterators"][it]["child_iterators"]:
            edge_index.append([it_index[it], it_index[child_it]])

        for child_comp in annotations["iterators"][it]["computations_list"]:
            edge_index.append([it_index[it], num_iterators + comp_index[child_comp]])
    node_feats = np.stack(
        [
            *[arr for arr in it_vector_dict.values()],
            *[arr for arr in comp_vector_dict.values()],
        ],
    )
    return node_feats, np.array(edge_index), it_index, comp_index


def apply_parallelization(iterator, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-6] = 1


def apply_reversal(iterator, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-5] = 1


def apply_unrolling(iterator, unrolling_factor, node_feats, it_index):
    index = it_index[iterator]
    node_feats[index][-4] = unrolling_factor


def apply_tiling(iterators, tile_sizes, node_feats, it_index):
    for it, tile in zip(iterators, tile_sizes):
        index = it_index[it]
        node_feats[index][-3] = tile


def apply_skewing(iterators, skewing_factors, node_feats, it_index):
    for it in iterators:
        index = it_index[it]
        node_feats[index][-2:] = skewing_factors


def apply_interchange(iterators, edge_index, it_index):
    it1, it2 = it_index[iterators[0]], it_index[iterators[1]]
    for edge in edge_index:
        if edge[0] == it1:
            edge[0] = it2
        elif edge[0] == it2:
            edge[0] = it1
        if edge[1] == it1:
            edge[1] = it2
        elif edge[1] == it2:
            edge[1] = it1


def focus_on_iterators(iterators, node_feats, it_index):
    # We reset the value for all the nodes
    node_feats[: len(it_index), -9:-8] = 0
    # We focus on the branches' iterators
    for it in iterators:
        index = it_index[it]
        node_feats[index][-9] = 1
    return node_feats