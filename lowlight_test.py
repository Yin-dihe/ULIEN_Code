import torch
import torchvision
import torch.optim
import os
import numpy as np
from PIL import Image
import glob
import model

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    ULIEN_net = model.ULIEN().cuda()
    ULIEN_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))

    r, g, b = data_lowlight[0][0] + 1, data_lowlight[0][1] + 1, data_lowlight[0][2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b)
    A_gray = torch.unsqueeze(A_gray, 0)
    A_gray = torch.unsqueeze(A_gray, 0)

    x, enhanced_image, output = ULIEN_net(data_lowlight, A_gray)

    image_path = image_path.replace('test_data', 'result')

    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image, result_path)



if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'data/test_data/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                # image = image
                print(image)
                lowlight(image)
