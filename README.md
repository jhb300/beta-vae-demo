# Disentangled Variational Autoencoder demo
A demonstration of the capabilities of beta-VAE to disentangle data-generative factors. This demonstration provides a pre-trained disentangled VAE model and an interface that allows a user to alter the latent variables intuitively in order to generate handwritten digits according based on human-interpretable data-generative factors.

## Getting Started
First, install the necessary requirements using 
```bash
pip install -r requirements.txt
```.
The package was developed using Python 3.9.6.

<br>
There are two ways to use this:
1. Jupyter Notebook: Run all cells in the demo.ipynb notebook and start by setting a certain number using the first slider.
2. Streamlit app: Run the streamlit app using ```streamlit run demo.py```. The web-app should open automatically.


## References
- Dupont, Emilien. "Learning disentangled joint continuous and discrete representations." Advances in neural information processing systems 31 (2018).
- JointVAE implementation: https://github.com/Schlumberger/joint-vae/tree/master, lastest retrieval 02nd Feb 2024
- Higgins, Irina, et al. "Early visual concept learning with unsupervised deep learning." arXiv preprint arXiv:1606.05579 (2016).
