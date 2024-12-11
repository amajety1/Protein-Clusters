# Protein Graph Embedding and Clustering

This repository contains a pipeline for graph embedding and clustering, designed to work with protein interaction networks or similar datasets represented as `.graphml` files. The pipeline extracts graph-level embeddings using techniques like DeepWalk and Graph2Vec, and applies clustering methods to uncover patterns or groupings within the data.

## Files in the Repository

1. **`Protein-Graph.ipynb`**
   - A Jupyter Notebook implementing the pipeline.
   - Reads `.graphml` files from a specified directory.
   - Generates embeddings using:
     - **DeepWalk**: Random walk-based embeddings combined with Word2Vec.
     - **Graph2Vec**: Unsupervised graph embedding technique.
   - Performs clustering on the learned graph embeddings.

2. **`LUT-48sample-ASTvalues7.csv`**
   - A sample CSV file containing clinical or biological measurements associated with the graphs.
   - Can be used to evaluate clustering results against known metadata or labels.

## Installation and Dependencies

The code is written in Python and requires the following libraries:

- `numpy`
- `pandas`
- `networkx`
- `gensim`
- `karateclub`
- `scikit-learn`

You can install the dependencies with:

```bash
pip install numpy pandas networkx gensim karateclub scikit-learn
```

## Workflow Overview

1. **Graph Reading**
   - Loads `.graphml` files from the specified directory.
   - Each file represents a graph (e.g., a protein-protein interaction network).

2. **Random Walks**
   - Generates node sequences using random walks (weighted or unweighted).

3. **Embedding Generation**
   - **DeepWalk**: Trains Word2Vec on random walks to generate node embeddings, then aggregates these into graph embeddings.
   - **Graph2Vec**: Learns graph-level embeddings directly using the Weisfeiler-Lehman graph kernel.

4. **Clustering**
   - Computes a similarity matrix using cosine similarity of embeddings.
   - Applies KMeans clustering to group graphs.

5. **Evaluation**
   - Results can be compared with the data in the `LUT-48sample-ASTvalues7.csv` file.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/amajety1/Protein-Clusters.git
   cd Protein-Clusters
   ```

2. Place your `.graphml` files in the root directory of the project.

3. Open `Protein-Graph.ipynb` in a Jupyter Notebook environment (e.g., Google Colab).

4. Modify the parameters as needed:
   - `dimensions`: Size of the embedding vector (default: 128).
   - `walk_length`: Length of random walks (default: 5).
   - `num_walks`: Number of random walks per node (default: 20).

5. Run the notebook to:
   - Generate graph embeddings.
   - Perform clustering.
   - Save results.

## Outputs

- Graph embeddings are saved as `.npy` files in the project directory.
- Clustering results are displayed in the notebook.

## Key Functions

- **`readnetworks()`**: Reads `.graphml` files into a list of NetworkX graph objects.
- **`random_walk()` and `random_walk_weighted()`**: Perform unweighted and weighted random walks.
- **`generate_deepwalk_embeddings()`**: Creates graph embeddings using DeepWalk.
- **`learn_embeddingGV()`**: Creates graph embeddings using Graph2Vec.
- **`learn_embeddingDW()`**: Learns embeddings for all graphs in the dataset.

## Example

To generate DeepWalk embeddings and perform clustering:

```python
GL, patientid = readnetworks()
dimensions = 128
walk_length = 5
num_walks = 20

# Learn embeddings
embeddings = learn_embeddingDW(GL, dimensions, walk_length, num_walks)

# Perform clustering
from sklearn.cluster import KMeans
num_clusters = 5
similarity_matrix = cosine_similarity(embeddings)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(similarity_matrix)
```

## Future Improvements

- Explore alternative graph embedding techniques.
- Optimize random walk parameters for better embeddings.
- Compare clustering results with additional evaluation metrics.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


