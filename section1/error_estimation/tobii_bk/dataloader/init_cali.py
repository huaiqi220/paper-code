

'''
 为每一个id生成一个8维cali向量，目前策略是随机生成，然后存到字典里

'''

import numpy as np
import json

# Step 1: Generate unique random vectors
def generate_unique_vectors(num_ids, vector_dim):
    vectors = {}
    while len(vectors) < num_ids:
        vec = np.random.randn(vector_dim)
        vec_tuple = tuple(vec)  # Convert to tuple to make it hashable
        if vec_tuple not in vectors:
            vectors[vec_tuple] = vec
    
    return {i: vec for i, vec in enumerate(vectors.values())}

# Step 2: Store the vectors in a JSON file (or use .npz for Numpy format)
def save_vectors_to_file(vectors, filename):
    with open(filename, 'w') as f:
        json.dump({str(k): v.tolist() for k, v in vectors.items()}, f)

# Step 3: Load the vectors from the file
def load_vectors_from_file(filename):
    with open(filename, 'r') as f:
        vectors = json.load(f)
    return {int(k): np.array(v) for k, v in vectors.items()}

# Example usage
# num_ids = 4000
# vector_dim = 8
# filename = "calibration_vectors.json"

# # Generate and save vectors
# vectors = generate_unique_vectors(num_ids, vector_dim)
# save_vectors_to_file(vectors, filename)

# # Load vectors for training
# loaded_vectors = load_vectors_from_file(filename)

# # Retrieve a vector by ID
# example_id = 42
# print(loaded_vectors[example_id])




