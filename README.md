# GAN Face Generator with Streamlit UI

This project demonstrates a Generative Adversarial Network (GAN) trained to generate human faces, presented with an interactive web UI built using Streamlit.

## Features
- Generates 64x64 pixel artificial faces.
- Trained on the CelebA dataset.
- Uses a DCGAN/WGAN-GP architecture implemented in PyTorch.
- Interactive Streamlit UI to generate and view faces.

## Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure PyTorch is installed according to your system/CUDA version from pytorch.org if not already present).*

3.  **Download the model (if using Git LFS and not cloned with LFS pull):**
    If `gan_generator_model.pth` is tracked with Git LFS and wasn't automatically pulled:
    ```bash
    git lfs pull
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app_streamlit.py
    ```
    Then open the provided URL in your browser.

## Model
The GAN generator model (`gan_generator_model.pth`) was trained for 55 epochs.
The architecture is based on the **Deep Convolutional Generative Adversarial Network (DCGAN)**, utilizing transposed convolutions in the generator to upsample noise into images and standard convolutions in the discriminator. Binary Cross-Entropy (BCE) loss was used for training.


