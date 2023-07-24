from PIL import Image
import torch

diseases = {'akiec': "atinic keratoses and intraepithelial carcinoma / Bowen’s disease ",
            'bcc': 'basal cell carcinoma',
            'bkl': "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
            'df': 'dermatofibroma',
            'mel': 'melanoma',
            'nv': 'melanocytic nevi',
            'vasc': 'vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)'}

def make_inference(path, model, transform, device, dataset):
    labelDecoder = {value: key for key, value in dataset.label_encoder.items()}
    
    my_image = Image.open(path).convert("RGB")
    my_image = transform(my_image)
    my_image = my_image.unsqueeze(0) 

    model.eval()
    with torch.no_grad():
        logits = model.forward(my_image.to(device))

    probs = torch.nn.functional.softmax(logits, dim=1)
    predTest = torch.argmax(probs,1)

    print(f'Diagnosis: {diseases.get(labelDecoder.get(predTest.item()))}') ## same value in all cases
    print(f'with confidence {torch.max(probs)*100}')
    #print(probs)
