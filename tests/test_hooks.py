import pytest 

import os 
os.chdir('../..')

import torch 
from softstairs_qat.core import SoftStairsQuantizer 
from softstairs_qat.wrappers import QuantizationConfig 

def test_hooks():

    test_model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.Softmax())
    quantizer = SoftStairsQuantizer(test_model, QuantizationConfig(n_bits=4, t_scheduler_strategy='linear', t_start=.3, type='standard'))

    with torch.no_grad():
        X = torch.randn(1, 10) 
        W = test_model[0].weight.clone() 

    opt = torch.optim.Adam(test_model.parameters())
    assert W.grad is None
    test_model.train()
    test_model(X).mean().backward()

    assert '0.weight_orig' in [n[0] for n in test_model.named_parameters()] 
    assert '0.weight' in [n[0] for n in test_model.named_buffers()] 
    assert test_model[0].weight_orig.grad is not None
    assert test_model[0].weight.grad is None

    opt.zero_grad()
    quantizer.fake_quantize()
    test_model.train()
    test_model(X).mean().backward()

    assert test_model[0].weight_orig.grad is None
    assert test_model[0].weight.grad is None

    quantizer.activate_hooks()
    opt.zero_grad()
    test_model(X).mean().backward()
    assert test_model[0].weight_orig.grad is not None
    assert test_model[0].weight.grad is None