from PIL import Image
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
from torch.autograd import Function


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable  

def loop_iterable_test(interable):
    yield from interable
    yield from interable
    
class GrayscaleToRgb:
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradCam(nn.Module):
    def __init__(self, model, module, layer):
        super().__init__()
        self.model = model
        self.module = module
        self.layer = layer
        self.register_hooks()

    def register_hooks(self):
        for modue_name, module in self.model._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):
        outs = self.model(input)
        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]
		
        # 가장 큰 값을 가지는 것을 target index 로 사용 
        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)
        
        # [128, 1, 1, 1]
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
        # [128, 2, 3, 2] * [128, 1, 1, 1]
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()
        out = torch.relu(out) / torch.max(out)
        #out = nn.Upsample(out.unsqueeze(0).unsqueeze(0), mode='trilinear', size = [193, 229, 193])
        out = F.interpolate(out.unsqueeze(0).unsqueeze(0), [193, 229, 193], mode='trilinear')
        #out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [193, 229, 193])
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])
