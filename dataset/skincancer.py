from torch.utils.data import Dataset
from PIL import Image

root_path = "data/Skin Cancer/Skin Cancer"

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.x = df["image_id"].to_numpy()
        self.y = df["dx"].to_numpy()
        self.label_encoder = {v: i for i, v in enumerate(df["dx"].unique())}
        self.transform = transform
        self.n_samples = df.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        image = Image.open(f'{root_path}/{self.x[index]}.jpg').convert("RGB")
        #image = torchvision.io.read_image(f'{root_path}/{self.x[index]}.jpg')
        label = self.label_encoder.get(self.y[index])
        if self.transform:
            image = self.transform(image)
        return image, label
        