# test_cuda.py
import torch

def main():
    print("Torch version           :", torch.__version__)
    print("CUDA available          :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA toolkit (torch)    :", torch.version.cuda)
        print("GPU count               :", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i} name        : {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability     : {torch.cuda.get_device_capability(i)}")
        # Test simple tensor operation on GPU
        x = torch.randn(3, 3).cuda()
        y = x * 2
        print("Tensor operation result :", y)
    else:
        print("Aucun GPU CUDA détecté.")

if __name__ == "__main__":
    main()
