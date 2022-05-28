from fastapi import FastAPI, UploadFile,File
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models,transforms
import io
from PIL import Image
##### ML model section
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def conv3x3(in_planes, out_planes, stride= 1, dilation = 1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride= 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    mul = 1
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.conv2 = conv3x3(out_planes, out_planes, 1)
        self.relu = nn.ReLU()
        self.bn1   = nn.BatchNorm2d(out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out  = self.conv1(x)
        out  = self.bn1(out)
        out  = self.relu(out)
        out  = self.conv2(out)
        out  = self.bn2(out)
        out += self.shortcut(x)
        out  = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()

        # 7x7, 64 channels, stride 2 in paper
        self.in_planes = 64 

        # RGB channel -> 64 channels
        self.conv    = nn.Conv2d(3, self.in_planes, kernel_size = 7, stride = 2, padding = 3)
        self.bn      = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.relu = nn.ReLU()

        _layers = []
        outputs, strides = [64, 128, 256, 512], [1, 2, 2, 2]
        for i in range(4):
            _layers.append(self._make_layer(block, outputs[i], num_blocks[i], stride=strides[i]))
        self.layers = nn.Sequential(*_layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(512 * block.mul, num_classes)

    def _make_layer(self, block, out_planes, num_block, stride):
        layers  = [ block(self.in_planes, out_planes, stride) ]
        self.in_planes = block.mul * out_planes
        for i in range(num_block - 1):
            layers.append(block(self.in_planes, out_planes, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layers(out) # layer 1 -> 2 -> 3 -> 4
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
PATH = './show_model.pth'
checkpoint = torch.load(PATH,map_location=device)
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
#transfer learning outputs를 3개로 바꿈
model.fc = nn.Linear(num_features,5)
model = model.to(device)
model.eval()
model.load_state_dict(checkpoint['downloaded_model_state_dict'])
c_model = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=5)
c_model.load_state_dict(checkpoint['custom_model_state_dict'])
c_model.eval()
transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
c_optimizer = optim.SGD(c_model.parameters(),lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()
#########fastapi section
def get_preds(image):
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.squeeze(0)
        outputs = F.softmax(outputs,dim=0)
        print(outputs.shape)
    
    return outputs.tolist()
def c_get_preds(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
    return outputs.tolist()

app = FastAPI()

@app.get('/')
def index():
    content = '''
    <body>
    <h3>upload image</h3><form action="/images/" method="post" enctype="multipart/form-data"><input name="image" type="file">
    <input type="submit"></form>
    </body>'''
    return HTMLResponse(content=content)

@app.post('/images')
async def create_upload_file(image:UploadFile = File(...)):
    
    image = await image.read()
    image_bytes = Image.open(io.BytesIO(image))
    result = get_preds(image_bytes)
    result = [round(el*100) for el in result]
    return {'result':result}
