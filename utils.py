from PIL import Image
from io import BytesIO
import numpy as np

import torch
import requests
from torchvision import transforms, models


def load_image(img_path, max_size=400, shape=None):
    '''
    Load local or online image, the transform includes:
    1. resize
    2. convet to tensor
    3. normalize RBG channel
    4. discard the alpha channel if exist
    5. add a batch dimension

    Parameters:
        img_path (str): local path or url of the image.
        max_size (int): the maximum size of the image.
        shape    (tuple): the shape (h,w) of the image.

    Return:
        image    (tensor): the tensor of the processed image.
    '''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
            
    in_transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def load_rand_image(tensor):
    '''
    Initialize an randomized image, with the shape of the input image. 

    Parameter:
        tensor  (tensor): image that provides the shape of the output.

    Returns:
        The generated image.
    '''
    return torch.Tensor(np.random.normal(0.5, 0.5, size=tensor.size()))


def im_convert(tensor):
    """
    Display a tensor as an image. 

    Parameter:
        tensor  (tensor): image in tensor dtype.

    Return:
        image   (numpy array)
    """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """
    Run an image forward through a model and get the features for 
    a set of layers. Default layers are for VGGNet matching Gatys et al (2016)

    Parameters:
        image    (tensor): image to be processed in tensor format.
        model    (torch.nn.Sequential): model used to extract style.
        layers   (dict): a dict which specifies the index and name of the layers 
                        to be extracted.

    Return:
        features (dict): a dictionary contains the processed tensors of the image 
                        in the correspoding layers.
    """
    if layers is None:
        layers = {
                    '0': 'conv1_1',
                    '5' : 'conv2_1',
                    '10': 'conv3_1',
                    '19': 'conv4_1',
                    '28': 'conv5_1'
                }
            
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
                    
    return features


def gram_matrix(tensor):
    """
    Calculate the Gram Matrix of a given tensor.

    Parameters:
        tensor (tensor): the tensor to be calculated.

    Return:
        gram   (tensor): the corresponding gram matrix.
    """
    
    gram = None
    batch_size, d, h, w = tensor.size()
    assert batch_size == 1, print('sth wrong')
    x = tensor.view(d, -1)
    gram = torch.mm(x, torch.transpose(x, 0, 1))
    
    return gram


def PCA(embedding, n_components):
    """
    Principle component analysis.

    Parameters:
        embedding   (Array of list):The embeddings to be processed.
        n_components(int):          The number of features generated.
    Return:
        Principle components.
    """
    x = pd.DataFrame(embedding)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)


def HDBSCAN(embedding, min_cluster_size, min_samples, alpha):
    """
    HDBSCAN Clustering.

    Parameters:
        embedding        (Array of list): The embeddings to be processed.
        min_cluster_size (int):           The minmum number of observations that could form a cluster.
        min_samples      (int):           The distance for group splitting.
        alpha   
    Return:
        Cluster object.
    """
    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, \
        min_samples=min_samples, alpha=alpha).fit(embedding)
    return clusters