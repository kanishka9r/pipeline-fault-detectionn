import torch.nn.functional as F
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.forward_hook
        )
        target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):
        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()
        output = self.model(x)
        pred_class = output.argmax(dim=1)
        score = output[
            0,
            pred_class.item()
        ]
        score.backward()
        weights = torch.mean(
            self.gradients,
            dim=2,
            keepdim=True
        )
        cam = torch.sum(
            weights * self.activations,
            dim=1
        )

        cam = F.relu(cam)
        cam = cam.squeeze()
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max()- cam.min()+ 1e-8)

        return cam, pred_class.item()

