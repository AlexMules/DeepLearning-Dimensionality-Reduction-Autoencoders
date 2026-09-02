# 🔤 EMNIST Letters: Dimensionality Reduction, Classification & Autoencoders

This repository contains the implementation and documentation for an advanced Deep Learning and Machine Learning project. The project explores an image-based dataset, applying dimensionality reduction techniques (PCA, t-SNE, UMAP), hyperparameter optimization for classification networks, and generative Autoencoder architectures.
<br><br>
## 📊 1. Dataset and Preprocessing
The project uses the **EMNIST Letters** dataset, which extends the classic MNIST to the English alphabet (A-Z).
* **Structure:** The dataset contains grayscale images of handwritten letters, each with a 28x28 pixel resolution, which have been flattened into a 784-feature vector. 
* **Preprocessing:** 
  * Pixel values were scaled (normalized) by dividing by 255.0, bringing them strictly into the [0, 1] range.
  * The labels, initially numbered 1 to 26, were decremented by one to fit the [0, 25] range, making them compatible with standard 0-indexed Python arrays.
  * To correctly visualize the geometric shape, the image matrices were transposed (`.T`), correcting the rotation and mirroring artifacts stemming from the original NIST digitization process.
<br><br>
## 📉 2. Dimensionality Reduction & Manifold Learning
We analyzed how different algorithms can extract and project information from a high-dimensional space (784D) into a 2D (or 3D) space.
* **Principal Component Analysis (PCA):** We applied the linear model described by the formula $Z=XW$ to evaluate the explained variance and reduce the number of components while retaining critical information.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear reduction was performed, optimizing hyperparameters such as learning rate and perplexity.
* **UMAP (Uniform Manifold Approximation and Projection):** We applied the UMAP method, adjusting specific parameters (`n_neighbors`, `min_dist`) to preserve global topology and create distinct clusters.
<br><br>
## 🧠 3. Optimization and Letter Classification
A neural network (MLP) dedicated to image classification was implemented.
* The architecture underwent a rigorous hyperparameter optimization process using `Grid Search` / `Randomized Search` techniques.
* Performance was evaluated through specific metrics and visualizations of loss and accuracy curves, proving the network's ability to accurately distinguish the 26 letter classes.
<br><br>
## 🧬 4. Autoencoder and Latent Space Analysis
The most complex component of the project involved training an Encoder-Decoder network using MSE as the loss function: $L(x,\hat{x})=||x-\hat{x}||^{2}$.
* **Optimal Architecture (32D):** Consisted of successive Dense layers for compression (784 -> 128 -> 64 -> 32) and decompression (32 -> 64 -> 128 -> 784). 
* **Loss Analysis (Lossy Compression):** Reconstruction through the 32D latent space achieved a loss function of 0.0132, corresponding to a mean pixel deviation (RMSE) of approximately 11.49% and a high structural fidelity of 88.51%.
* **The Bottleneck Effect (2D Experiment):** Dimensionality reduction was forced down to just 2 neurons, which severely degraded quality. The loss function significantly increased to 0.0576, raising the per-pixel deviation to nearly 24% and dropping fidelity to 76.09%.
* **Autoencoder + UMAP Synergy:** Applying the UMAP algorithm exclusively on the 32D latent space extracted from the Autoencoder resulted in dense and isolated geometric class filaments, demonstrating vastly superior topological separation compared to applying PCA or UMAP on the raw pixels.
<br><br>
## 🛠️ Technologies Used
* **Language:** Python 3
* **Machine Learning & Deep Learning:** TensorFlow / Keras, Scikit-Learn, UMAP-learn
* **Data Manipulation & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
* **Data Source:** Kaggle API (`kaggle datasets download -d crawford/emnist`)
<br><br>
## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/AlexMules/DeepLearning-Dimensionality-Reduction-Autoencoders.git
   ```

2. Ensure you have the **`kaggle.json`** file configured to download the EMNIST dataset via API.
3. Install the required dependencies:
   ``` bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow umap-learn kaggle
   ```
4. Run the script/notebook directly in Google Colab or a local Jupyter environment.
