import torch

def get_generator():
    with torch.inference_mode():
        for i in range(10):
            yield i

def foo():
    gen = get_generator()
    next(gen)
    print(torch.is_inference_mode_enabled())

print(torch.is_inference_mode_enabled())
foo()
print(torch.is_inference_mode_enabled())
