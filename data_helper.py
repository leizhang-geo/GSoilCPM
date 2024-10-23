import os
import numpy as np
import os.path
import torch.utils.data
import utils
import torchvision.transforms as transforms


def default_image_loader(path):
    return utils.load_pickle(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_data_band, x_data_topo, x_data_climate, x_data_vege, x_data_bedrock, y_data, data_index=None, transform=None, shuffle=False):
        if data_index is None:
            self.x_band = x_data_band
            self.x_topo = x_data_topo
            self.x_climate = x_data_climate
            self.x_vege = x_data_vege
            self.x_bedrock = x_data_bedrock
            self.y = y_data
        else:
            self.x_band = x_data_band[data_index]
            self.x_topo = x_data_topo[data_index]
            self.x_climate = x_data_climate[data_index]
            self.x_vege = x_data_vege[data_index]
            self.x_bedrock = x_data_bedrock[data_index]
            self.y = y_data[data_index]
        
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_band_one = self.x_band[index]
        x_topo_one = self.x_topo[index]
        x_climate_one = self.x_climate[index]
        x_vege_one = self.x_vege[index]
        x_bedrock_one = self.x_bedrock[index]
        y_one = self.y[index]
        x_band_one = torch.tensor(x_band_one, dtype=torch.float32)
        x_topo_one = torch.tensor(x_topo_one, dtype=torch.float32)
        x_climate_one = torch.tensor(x_climate_one, dtype=torch.float32)
        x_vege_one = torch.tensor(x_vege_one, dtype=torch.float32)
        x_bedrock_one = torch.tensor(x_bedrock_one, dtype=torch.float32)
        y_one = torch.tensor(y_one, dtype=torch.float32)
        if self.transform is not None:
            x_band_one = self.transform(x_band_one)
            x_topo_one = self.transform(x_topo_one)
            x_climate_one = self.transform(x_climate_one)
            x_vege_one = self.transform(x_vege_one)
            x_bedrock_one = self.transform(x_bedrock_one)
        return x_band_one, x_topo_one, x_climate_one, x_vege_one, x_bedrock_one, y_one

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.y)).tolist())
        else:
            return iter(range(len(self.y)))
        

class DatasetX(torch.utils.data.Dataset):
    def __init__(self, x_data_band, x_data_topo, x_data_climate, x_data_vege, x_data_bedrock, data_index=None, transform=None, shuffle=False):
        if data_index is None:
            self.x_band = x_data_band
            self.x_topo = x_data_topo
            self.x_climate = x_data_climate
            self.x_vege = x_data_vege
            self.x_bedrock = x_data_bedrock
        else:
            self.x_band = x_data_band[data_index]
            self.x_topo = x_data_topo[data_index]
            self.x_climate = x_data_climate[data_index]
            self.x_vege = x_data_vege[data_index]
            self.x_bedrock = x_data_bedrock[data_index]
        
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_band_one = self.x_band[index]
        x_topo_one = self.x_topo[index]
        x_climate_one = self.x_climate[index]
        x_vege_one = self.x_vege[index]
        x_bedrock_one = self.x_bedrock[index]
        x_band_one = torch.tensor(x_band_one, dtype=torch.float32)
        x_topo_one = torch.tensor(x_topo_one, dtype=torch.float32)
        x_climate_one = torch.tensor(x_climate_one, dtype=torch.float32)
        x_vege_one = torch.tensor(x_vege_one, dtype=torch.float32)
        x_bedrock_one = torch.tensor(x_bedrock_one, dtype=torch.float32)
        if self.transform is not None:
            x_band_one = self.transform(x_band_one)
            x_topo_one = self.transform(x_topo_one)
            x_climate_one = self.transform(x_climate_one)
            x_vege_one = self.transform(x_vege_one)
            x_bedrock_one = self.transform(x_bedrock_one)
        return x_band_one, x_topo_one, x_climate_one, x_vege_one, x_bedrock_one

    def __len__(self):
        return len(self.x_band)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.x_band)).tolist())
        else:
            return iter(range(len(self.x_band)))
