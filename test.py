import torch
from network_architecture import ALIXEncoder1, ParameterizedReg, \
    LocalSignalMixing, StackPolicyNetwork
model = torch.load('trained model/1/50k/50k.pt', map_location='cpu')
aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True),
                               parameter_init=0.5, param_grad_fn='alix_param_grad',
                               param_grad_fn_args=[3, 0.535, 1e-20])
encoder = ALIXEncoder1((140, 140, 4), aug=aug)
print(1)
