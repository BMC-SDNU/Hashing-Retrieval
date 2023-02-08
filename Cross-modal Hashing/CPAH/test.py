import torch
if __name__ == '__main__':
    conv1 = torch.nn.Conv1d(in_channels=1, out_channels = 512, kernel_size = 512, stride=1)
    input = torch.ones(1000, 512).unsqueeze(1)
    out = conv1(input)
    print(out.shape)
    pass