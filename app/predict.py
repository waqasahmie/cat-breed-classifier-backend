import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Setup
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (sorted)
class_names = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "BritishShorthair", "EgyptianMau",
    "MaineCoon", "Persian", "Ragdoll", "RussianBlue", "Siamese", "Sphynx"
]

# Load ResNet50 and define custom classifier head
class ClassifierHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model and load weights
model = resnet50(weights=None)
model.fc = ClassifierHead(2048, 1024, len(class_names))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()

    threshold = 0.05  # 5%
    top_probs, top_indices = probs.sort(descending=True)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        if prob.item() < threshold or len(results) == 2:
            break
        breed = class_names[idx]
        confidence = prob.item() * 100
        results.append((breed, confidence))

    return results
