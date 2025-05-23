from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine

from edgar_funcs.rag.vectorize import TextChunksWithEmbedding


def top_adjacent_chunks(relevance_scores) -> list[str]:
    """
    select top 3 chunks, return the adjacent chunks
    """
    top_chunks = [chunk_num for chunk_num, _, _, _ in relevance_scores[:3]]
    selected_chunks = []
    if top_chunks:
        selected_chunks.append(top_chunks[0])
        if len(top_chunks) > 1:
            if abs(top_chunks[0] - top_chunks[1]) == 1:
                selected_chunks.append(top_chunks[1])
        if len(top_chunks) > 2:
            if abs(top_chunks[0] - top_chunks[2]) == 1:
                selected_chunks.append(top_chunks[2])

        selected_chunks = list(set(selected_chunks))
        selected_chunks.sort()
        return selected_chunks

    return []


def top_chunks(relevance_scores, top_k: int) -> list[str]:
    """
    just return top chunks, sorted, regardless if they're adjacent or not
    """
    top_chunks = [chunk_num for chunk_num, _, _, _ in relevance_scores[:top_k]]
    top_chunks.sort()
    return top_chunks


def relevance_by_distance(chunk_distances):
    """
    Calculate relevance scores for chunks based on their distances.

    This function takes a dictionary of chunk distances, computes the relevance
    score for each chunk, and ranks the chunks based on their scores and tie-breaking rules.

    Args:
        chunk_distances (dict):
            A dictionary where:
            - Keys are chunk numbers (int).
            - Values are lists of distances (float) associated with that chunk.

    Returns:
        list of tuple:
            A sorted list of tuples, where each tuple contains:
            - chunk_num (int): The chunk number.
            - score (float): The calculated relevance score (higher is better).
            - avg_distance (float): The average distance for the chunk.
            - min_distance (float): The minimum distance in the chunk (used for tie-breaking).

            The list is sorted in descending order by score.
            If scores are tied, the chunks are sorted in ascending order by min_distance.

    Example:
        >>> chunk_distances = {
        ...     159: [0.282, 0.345],
        ...     158: [0.291],
        ...     444: [0.311, 0.316],
        ... }
        >>> calculate_relevance_2(chunk_distances)
        [
            (159, 0.747, 0.3135, 0.282),
            (158, 0.774, 0.291, 0.291),
            (444, 0.751, 0.3135, 0.311),
        ]
    """  # noqa E501
    # Calculate metrics for each chunk
    scores = []
    for chunk_num, distances in chunk_distances.items():
        avg_distance = sum(distances) / len(distances)
        min_distance = min(distances)  # For tie-breaking
        score = 1 / (1 + avg_distance)  # Modified scoring function
        scores.append((chunk_num, score, avg_distance, min_distance))

    # Rank by score (descending), then by min_distance (ascending)
    scores.sort(key=lambda x: (-x[1], x[3]))

    # Return the ranked chunks
    return scores


def relevance_by_appearance(chunk_distances):
    """
    Calculate relevance scores for chunks based on their frequency and distances.

    This function computes a relevance score for each chunk using its frequency of
    appearance and the average distance. Chunks are ranked by their relevance scores
    in descending order.

    Args:
        chunk_distances (dict):
            A dictionary where:
            - Keys are chunk numbers (int).
            - Values are lists of distances (float) associated with that chunk.

    Returns:
        list of tuple:
            A sorted list of tuples, where each tuple contains:
            - chunk_num (int): The chunk number.
            - frequency (int): The number of distances (frequency) for the chunk.
            - avg_distance (float): The average distance for the chunk.
            - score (float): The calculated relevance score (higher is better).

            The list is sorted in descending order by the relevance score.

    Example:
        >>> chunk_distances = {
        ...     159: [0.282, 0.345],
        ...     158: [0.291],
        ...     444: [0.311, 0.316],
        ... }
        >>> relevance_by_appearance(chunk_distances)
        [
            (159, 2, 0.3135, 1.274),
            (444, 2, 0.3135, 1.274),
            (158, 1, 0.291, 0.774),
        ]
    """  # noqa E501
    relevance_scores = []
    for chunk_num, distances in chunk_distances.items():
        frequency = len(distances)  # Number of distances for the chunk
        avg_distance = sum(distances) / frequency  # Average distance
        score = frequency / (1 + avg_distance)  # Relevance score
        relevance_scores.append((chunk_num, frequency, avg_distance, score))

    # Sort by score in descending order
    relevance_scores.sort(key=lambda x: x[3], reverse=True)
    return relevance_scores


def gather_chunk_distances(results: list[dict]) -> dict:
    chunk_distances = {}
    for row in results:
        chunk_num = row["chunk_num"]
        distance = row["distance"]
        if chunk_num not in chunk_distances:
            chunk_distances[chunk_num] = []
        chunk_distances[chunk_num].append(distance)

    return chunk_distances


def nearest_chunks(
    queries: list[list[float]],
    contents: list[list[float]],
    top_k: int,
    filtered_chunk_nums: list[int] = [],
):
    """
    Find nearest chunks to query embeddings based on cosine distance.

    Args:
        queries: List of query embedding vectors
        contents: List of content embedding vectors
        top_k: Number of top results to return
        filtered_indices: Optional list of indices to restrict the search to.
                          If provided, only these indices from contents will be used.
                          This allows pre-filtering using methods like BM25.

    Returns:
        List of dictionaries with query_idx, chunk_num, and distance
    """
    distances = []

    # If filtered_indices is empty, process all contents
    indices_to_process = (
        range(len(contents)) if not filtered_chunk_nums else filtered_chunk_nums
    )

    for chunk_num in indices_to_process:
        content_vec = contents[chunk_num]
        for query_idx, query_vec in enumerate(queries):
            distance = float(cosine(query_vec, content_vec))
            distances.append(
                {
                    "query_idx": query_idx,
                    "chunk_num": chunk_num,
                    "distance": distance,
                }
            )

    distances.sort(key=lambda x: x["distance"])
    return distances[:top_k]


def preprocess_text(text: str) -> list[str]:
    """
    Preprocess text for BM25 tokenization, with special handling for dollar amounts.
    This helps match dollar ranges regardless of formatting.
    """
    # Basic lowercase and split
    tokens = text.lower().split()

    # Process tokens to handle dollar amounts
    processed_tokens = []
    for token in tokens:
        # Keep original token
        processed_tokens.append(token)

        # Handle dollar signs and commas in numbers
        if "$" in token or "," in token:
            # Remove $ and commas from numbers
            stripped = token.replace("$", "").replace(",", "")
            if stripped.isdigit() or (
                stripped.replace("-", "").isdigit() and "-" in stripped
            ):
                processed_tokens.append(stripped)

    return processed_tokens


def filter_chunks_with_keywords(
    chunks: TextChunksWithEmbedding,
    keywords: list[str],
    top_k: int,
) -> list[int]:
    """
    Filter chunks using BM25 algorithm based on keywords.
    Returns the indices of the top k chunks.
    Handles special cases like dollar amounts with or without $ signs.
    """
    corpus = chunks.texts
    # Use custom tokenization that handles dollar amounts
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Preprocess the keywords with the same function
    tokenized_query = preprocess_text(" ".join(keywords))
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get indices of top k chunks by BM25 score
    top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:top_k]
    return top_indices
