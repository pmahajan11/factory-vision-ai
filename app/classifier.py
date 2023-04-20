import torch
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

if torch.cuda.is_available():
    print("cuda available")
    model.to('cuda')
else:
    print("cuda not available")

with open("app/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    return categories[top_catid[0]], round(top_prob[0].item(), 4)