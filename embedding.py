from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

vec=model.encode("The cat is on the mat.")

print(vec.shape)
print(vec[:5])

v1=model.encode("The cat is on the couch.")
v2=model.encode("The kitten is on the sofa.")

similarity = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
print(f"Similarity: {similarity:.4f}")