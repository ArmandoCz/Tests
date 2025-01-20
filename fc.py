import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("torch.log"),
        logging.StreamHandler()
    ]
)

logging.info(f"CUDA available? {torch.cuda.is_available()}")
logging.info(f"CUDA device:{torch.cuda.get_device_name(0) if torch.cuda.is_available else 'no hay GPU'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.tensor([1.0,2.0],requires_grad=True)
y = x**2
print(y)
y.backward(torch.tensor([1.0,1.0]))
print(y)
print(x.grad)

a = torch.tensor([1,2,3],device=device)
b = torch.tensor([[1,2],[3,4]],device=device,dtype=float)
c = b @ b.T

print(f"This is b:\n{b}")
print(f"This is b:\n{c}")
