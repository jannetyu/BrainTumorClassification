import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# class definition from jpynb

# defining my cnn model

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dynamically compute the correct number of input features for fc1
        self.fc1_input_features = self.get_fc1_input_features()

        # connect layers
        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        self.fc2 = nn.Linear(512, 2)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Feed through the fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def get_fc1_input_features(self):
    # Pass a dummy input through the convolutional layers to get the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)  # Update the input size to 256x256
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(x.size(0), -1).size(1)



# Load your trained model (make sure to define your model class first)
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = SimpleCNN()  # Replace with your actual class
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('/Users/janet/Desktop/2nd Year - UCLA/winter24/DSU Project/brain_tumor_detector.pth')

# Define a function to preprocess the image for your model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Match the size used during training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Apply the same normalization as training, assuming RGB
    ])
    return transform(image).unsqueeze(0)

# Define the main function for the Streamlit app
def main():
    st.title("CNN Image Classification with PyTorch")

    # User upload interface
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and predict
        preprocessed_image = preprocess_image(image)
        with torch.no_grad():
            prediction = model(preprocessed_image)
            probabilities = F.softmax(prediction, dim=1)
            _, predicted_class = torch.max(probabilities, 1)
        
        # Map the predicted class index to the corresponding string label
        predicted_label = 'yes' if predicted_class.item() == 1 else 'no'

        # Display the prediction
        st.write(f'Predicted class: {predicted_label}')

if __name__ == "__main__":
    main()
