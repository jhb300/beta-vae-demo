import torch
import numpy as np
from jointvae.models import VAE
from utils.load_model import load


class Visualizer():
    def __init__(self, parameter_path: str) -> None:
        """
        Visualizer that visualizes the output of the JointVAE decoder given a set
        of data-generative factors.
        """
        
        self.parameter_path = parameter_path
        self.model = self.load_model(parameter_path)

        # Configure model for inference.
        self.model.eval()

    def load_model(self, parameter_path: str) -> VAE:
        "Load JointVAE model parameters from given path."

        return load(parameter_path)

    def visualize(self, discrete_latent_parameters: list, continous_latent_parameters: list) -> np.ndarray:
        "Visualize using the self model and a set of parameters, i.e. decoder input."

        assert self.model != None, "No model has been defined yet to run the visualizations on."

        concatenated_parameters = torch.tensor([*continous_latent_parameters, *discrete_latent_parameters], dtype=torch.float)
        with torch.no_grad():
            img_tensor = self.model.decode(concatenated_parameters)
            clean_img_tensor = img_tensor.squeeze()
            
            # If it's a colored image.
            if len(clean_img_tensor) == 3:
                clean_img_tensor = clean_img_tensor.permute(1, 2, 0)
                print(img_tensor.shape)

        # Convert to image
        image_np = clean_img_tensor.detach().numpy()

        return image_np

