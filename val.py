from PIL import Image
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from VGG.VGG16 import VGG_
import VGG.VGG16

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = torch.device("cpu")
model = torch.load('VGG_.pt').to(device)  #可以自行选择模型比如VGG或resnet
# 加载图像
image_path = '../ship.jpg'  # 替换为你想检验的图像

img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
# 应用转换
img_tensor = transform(img).to(device)

# 如果需要，可以手动添加批次维度（例如，形状变为 [1, C, H, W]）
img_tensor = img_tensor.unsqueeze(0)
model.eval()
outputs = model(img_tensor)
_, predicted = torch.max(outputs.data, 1)

MAP = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
       9: 'truck'}
p = predicted.item()
print(MAP[p])
