# Disentangled Variational Autoencoder demo
Submission by: Andre Gilbert (4365546) and Jan Henrik Bertrand (8556462)
A demonstration of the capabilities of beta-VAE to disentangle data-generative factors. This demonstration provides a pre-trained disentangled VAE model and an interface that allows a user to alter the latent variables intuitively in order to generate handwritten digits according based on human-interpretable data-generative factors.<br>
The demo is based on JointVAE, a Variational Autoencoder that has both continuous and discrete latent variables (cf. Dupont, 2018).

## Getting Started

First, install the necessary requirements using the following command:

```bash
pip install -r requirements.txt
```

The package was developed using Python 3.9.6.

<br>
There are two ways to use this:

1. <b>Jupyter Notebook:</b> Run all cells in the `demo.ipynb` notebook and start by setting the first slider.

2. <b>Streamlit App:</b> Run the streamlit app using the following command:
   ```bash
   streamlit run demo.py
   ```

The web-app should open automatically. Live version: https://beta-vae-demo.streamlit.app.

## References

- Dupont, Emilien. "Learning disentangled joint continuous and discrete representations." Advances in neural information processing systems 31 (2018).
- JointVAE implementation: https://github.com/Schlumberger/joint-vae/tree/master, lastest retrieval 02nd Feb 2024
- Higgins, Irina, et al. "Early visual concept learning with unsupervised deep learning." arXiv preprint arXiv:1606.05579 (2016).
