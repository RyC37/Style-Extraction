import torch
import torch.optim as optim
from torchvision.utils import save_image
from utils import * 

def recon_style(img_path, device, model, out_fname, style_weights, show_every=100, steps=800):

    model.to(device)
    style = load_image(img_path).to(device)
    style_features = get_features(style, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = load_rand_image(style).to(device).detach().requires_grad_(True)

    optimizer = optim.Adam([target], lr=0.01)

    for i in range(1, steps+1):
        target_features = get_features(target, model)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matrix(target_feature)
            style_feature = style_features[layer]
            style_gram = gram_matrix(style_feature)
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
            
        total_loss = style_loss
        # update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # export intermediate images and print the loss
        if i % show_every == 0:
            print('Total loss: ', total_loss.item())
            # res = im_convert(target)
            save_image(target, "./export/"+out_fname+"_%s.png" % i, normalize=True)
            # plt.imshow(im_convert(target))
            # plt.axis('off')
            # plt.savefig('./export/' + export_fname + str(int(i)/show_every) + '.png')

    return target, im_convert(target)
