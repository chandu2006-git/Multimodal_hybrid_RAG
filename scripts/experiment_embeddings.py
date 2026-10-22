
from sentence_transformers import SentenceTransformer

def main() -> None:
    # 1. Load the pretrained embedding model
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Prepare example sentences
    sentences = [
        "The neural network detects face morph attacks.",
        "The model identifies manipulated facial images.",
        "The weather will be sunny tomorrow.",
    ]

    # 3. Convert sentences into vectors
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
    )

    print("Number of sentences:", len(sentences))
    print("Embedding shape:", embeddings.shape)

    # 4. Calculate cosine similarities
    similarity_matrix = embeddings @ embeddings.T

    print("\nCosine similarity matrix:")
    for row in similarity_matrix:
        print([round(float(score), 3) for score in row])

    print("\nSentence A vs Sentence B:")
    print(round(float(similarity_matrix[0, 1]), 3))

    print("\nSentence A vs Sentence C:")
    print(round(float(similarity_matrix[0, 2]), 3))


if __name__ == "__main__":
    main()
