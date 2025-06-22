# Urdu Text Summarizer using TextRank

## Description

This project is an unsupervised extractive text summarizer specifically designed for the Urdu language. Given the scarcity of NLP resources for Urdu, this tool provides an effective way to condense lengthy documents into concise, relevant summaries. It leverages the **TextRank** algorithm, a graph-based ranking model inspired by Google's PageRank, to identify and extract the most significant sentences from a text. The entire process, from word embedding generation to sentence ranking, is implemented from scratch and does not require any pre-trained models.

The primary motivation is to bridge the resource gap for widely spoken but underrepresented languages in the NLP space, making information more accessible.

## How It Works

The summarizer follows a classic unsupervised, graph-based pipeline:

1.  **Text Preprocessing**: The input Urdu text is cleaned and tokenized into individual sentences using the `UrduHack` library.
2.  **Word Embedding**: A custom n-gram (bigram) language model is trained on the input corpus using **PyTorch** to generate 100-dimensional vector embeddings for each unique word in the text.
3.  **Sentence Vectorization**: Each sentence is converted into a vector by averaging the embeddings of the words it contains.
4.  **Similarity Matrix**: The **Cosine Similarity** between every pair of sentence vectors is calculated to determine how related they are. This creates a similarity matrix that represents the relationships between all sentences.
5.  **Graph-Based Ranking**: The similarity matrix is used as an adjacency matrix to create a graph, where sentences are nodes and similarity scores are edge weights. The **TextRank** algorithm (implemented via `NetworkX`'s PageRank) is then run on this graph to assign an importance score to each sentence.
6.  **Summary Generation**: The sentences are ranked based on their scores, and the top-N sentences are selected to form the final extractive summary.


*(Conceptual diagram of the project workflow)*

## Features

-   **Extractive Summarization**: Creates a summary by selecting the most important sentences from the original text.
-   **Unsupervised Approach**: Requires no labeled data or pre-trained models for summarization.
-   **Urdu Language Support**: Specifically tailored for Urdu text using `UrduHack` for robust preprocessing.
-   **Custom Word Embeddings**: Generates word vectors from scratch for the input corpus using a PyTorch-based n-gram model.
-   **TextRank Algorithm**: Implements the graph-based TextRank algorithm to rank sentences by importance.
-   **ROUGE Evaluation**: Includes code to evaluate the quality of the generated summary against a human-written reference using ROUGE metrics.

## Getting Started

Follow these instructions to get a copy of the project running on your local machine.

### Prerequisites

-   Python 3.9+
-   pip

The project relies on several Python libraries. You can install them using the provided `requirements.txt` file.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
  

## Usage

To run the summarizer and see the results:

1.  Make sure the input file `urduarticle.txt` and the stopwords file `kg_uswords.txt` are present in the root directory of the project.
2.  Open and run the Jupyter Notebook `code.ipynb` from top to bottom.

    ```sh
    jupyter notebook code.ipynb
    ```

The notebook will process the text, generate a 7-sentence summary, and print the ROUGE evaluation scores at the end.

## Evaluation & Results

The quality of the generated summary is measured against a human-written reference summary using the **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metric. The key scores from the evaluation are:

| Metric    | F1-Score | Precision | Recall |
| :-------- | :------: | :-------: | :----: |
| **ROUGE-1** |  0.550   |   0.581   | 0.536  |
| **ROUGE-2** |  **0.511**   |   0.519   | 0.504  |
| **ROUGE-L** |  0.550   |   0.581   | 0.536  |

The **ROUGE-2 F1-score of 0.511** indicates a good overlap of bigrams (two-word sequences) between the model-generated summary and the human reference, demonstrating the model's effectiveness.

## Project Files

-   `code.ipynb`: The main Jupyter Notebook containing all the code, from data preprocessing and word embedding generation to summarization and evaluation.
-   `urduarticle.txt`: An example Urdu text document used as input for summarization.
-   `kg_uswords.txt`: A list of Urdu stopwords used during text preprocessing.
-   `report.pdf`: The detailed project report explaining the methodology and motivation.

## Dependencies

-   [Python](https://www.python.org/)
-   [PyTorch](https://pytorch.org/)
-   [UrduHack](https://github.com/urduhack/urduhack)
-   [NetworkX](https://networkx.org/)
-   [scikit-learn](https://scikit-learn.org/stable/)
-   [NumPy](https://numpy.org/)
-   [Pandas](https://pandas.pydata.org/)
-   [rouge](https://pypi.org/project/rouge/)

