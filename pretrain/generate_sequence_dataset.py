import numpy as np
import pickle
import re

MAX_DEPTH = 5
max_accesses = 15

train_data_path = "./pretrain/data/train_data_sample_500-programs_60k-schedules.pkl"
val_data_path = "./pretrain/data/val_data_sample_125-programs_20k-schedules.pkl"

def pad_access_matrix(access_matrix):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix

def get_access_matrices(program_annotation):
    access_matrices = []

    computations_dict = program_annotation["computations"]
    computations_list = list(computations_dict.keys())
    for comp_idx, comp_name in enumerate(computations_list):
        comp_dict = computations_dict[comp_name]
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"]
            )
            access_matrices.append(read_access_matrix.flatten().tolist())
    return access_matrices

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

def get_sequence(comp_dict):
    sequence = []

    padded_write_matrix = pad_access_matrix(
        isl_to_write_matrix(comp_dict["write_access_relation"])
    )
    write_access_repr = [
        -1,
        comp_dict["write_buffer_id"] + 1
    ] + padded_write_matrix.flatten().tolist()
    
    print("write access repr: ", len(write_access_repr))

    sequence.append(write_access_repr)

    # Pad the read access matrix and add it to the representation 
    # read_accesses_repr = []
    for read_access_dict in comp_dict["accesses"]:
        read_access_matrix = pad_access_matrix(
            read_access_dict["access_matrix"]
        )
        read_access_repr = (
            [+read_access_dict["access_is_reduction"]]
            + [read_access_dict["buffer_id"] + 1]
            + read_access_matrix.flatten().tolist()
        )
        
        sequence.append(read_access_repr)
        print("read access repr: ", len(read_access_repr))

    access_repr_len = (MAX_DEPTH + 1) * (MAX_DEPTH + 2) + 1 + 1

    for i in range(max_accesses - len(comp_dict["accesses"])):
        sequence.append([0]*access_repr_len)
        
    print("sequence len: ", len(sequence), len(sequence[0]))    
    return sequence

def generate_datasets(data_path):
    dataset = []
    with open(data_path, "rb") as input_file:
        programs_dict = pickle.load(input_file)
        function_name_list = list(programs_dict.keys())
        func_cnt = 0
        for function_name in function_name_list:
            print("function ", func_cnt)
            func_cnt += 1
            comp_name_list = list(programs_dict[function_name]["program_annotation"]["computations"].keys())
            print("comp name list len: ", len(comp_name_list))
            for comp_name in comp_name_list:
                sequence = get_sequence(programs_dict[function_name]["program_annotation"]["computations"][comp_name])
                dataset.append(sequence)
    return dataset

if __name__ == "__main__":
    
    train_dataset = generate_datasets(train_data_path)
    val_dataset = generate_datasets(val_data_path)
    
    print("calculating number of sequences ... ")
    print("train dataset vector length: ", len(train_dataset[0]))
    print("val dataset vector length: ", len(val_dataset[0]))
    
    np.save("pretrain/data/sequence_dataset_train", train_dataset , allow_pickle = True)
    np.save("pretrain/data/sequence_dataset_val", val_dataset , allow_pickle = True)
    
    print("Finished generating sequence dataset")
    print("train dataset length: ", len(train_dataset))
    print("val dataset length: ", len(val_dataset))