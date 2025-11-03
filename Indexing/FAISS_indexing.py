import numpy as np
import faiss
import time

def build_faiss_index(features_dict):
    """Builds a FAISS index from the feature dictionary using cosine similarity."""
    image_paths = list(features_dict.keys())
    features = [features_dict[path][1] for path in image_paths]
    features = np.array(features).astype('float32')

    # Normalize features (for cosine similarity with inner product)
    faiss.normalize_L2(features)

    # Build FAISS index
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)

    # 🟢 Ensure timestamps are floats
    timestamps = [float(features_dict[p][0]) for p in image_paths]

    return index, image_paths, timestamps # return timestamps too

def create_filtered_index(original_index, filtered_indices):
    start = time.time()
    all_features = original_index.reconstruct_n(0, original_index.ntotal)
    filtered_features = all_features[filtered_indices]
    faiss.normalize_L2(filtered_features)

    temp_index = faiss.IndexFlatL2(filtered_features.shape[1])
    temp_index.add(filtered_features)

    print(f"Index filtering time: {(time.time() - start)*1000:.2f} ms")
    return temp_index, filtered_indices

def run_similarity_search(index, query_vector, top_k):
    start = time.time()
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, top_k)
    print(f"Similarity search time: {(time.time() - start)*1000:.2f} ms")
    return D, I
