
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from run import run_style_transfer
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([transforms.Resize(imsize),  # scale imported image
                             transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader('./data/picasso.jpg')  # [1, 3, 128, 128]
content_img = image_loader('./data/dancing.jpg')  # [1, 3, 128, 128]

assert style_img.size() == content_img.size(), 'we need to import style and content images of the same size'

unloader = transforms.ToPILImage()

plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# plt.figure()
# imshow(style_img, title='Style Image')
#
# plt.figure()
# imshow(content_img, title='Content Image')

######################################

input_img = content_img.clone()
# plt.figure()
# imshow(input_img, title='Input Image')

######################################

cnn = models.vgg19(pretrained=True).features.to(device).eval()

normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

output = run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img)
plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()
