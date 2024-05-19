import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision import models

def predict(image):
    print(type(image))
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    # Load model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("best_f1.pth"))
    model.eval()
    
    # Preprocess image
    valid_transform = transforms.Compose([
        # transforms.ToPILImage(),                # Convert the image to a PIL Image
        transforms.Resize((224, 224)), # Resize the image to final_size x final_size
        transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
        transforms.Normalize(                  # Normalize the image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_batch = valid_transform(image).unsqueeze(0)
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        output = torch.sigmoid(output).squeeze().item()
        if output > 0.5:
            predicted = 1
        else:
            predicted = 0
    
    int2label = {0: "cat", 1: "dog"}
    return int2label[predicted]

demo = gr.Interface(
    predict, 
    inputs="image", 
    outputs="label",
    title="Cats vs Dogs",
    description="This model predicts whether an image contains a cat or a dog.",
    examples = ["assets/7.jpg", "assets/44.jpg", "assets/82.jpg", "assets/83.jpg"]
)

demo.launch()