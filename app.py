import torchvision
from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the pre-trained PyTorch model
state_dict = torch.load('./resnet18-5c106cde.pth')

# Create an instance of the model and load the state dictionary
model = torchvision.models.resnet18()
model.load_state_dict(state_dict)
model.eval()

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize_image():
    print('checking image!')
    # Assuming the image file is sent in the 'image' field of the request
    image_file = request.files['image']
    print('Received image file:', image_file.filename)
    # Save the image to a temporary location
    image_path = './try-to-recognize.jpeg.jpg'
    image_file.save(image_path)

    # Load the image
    image = Image.open(image_path)

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 2)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    # Perform post-processing on the output and convert to a human-readable format
    result = 'Some recognition result'


    # Return the output list as part of the response
    response = {
        'result': result
    }
    return response


if __name__ == '__main__':
    app.debug = True
    app.run()
