from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import argparse
import os
from glob import glob
from models.blip import blip_decoder
import jsonlines

def get_args():
    parser = argparse.ArgumentParser(description="Obtain Hungginface dataset metadata used for stable diffusion training.")
    parser.add_argument("--data", type=str)
    args = parser.parse_args()
    return args

def load_demo_image(image_path, image_size, device):
    #image_path = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(image_path).convert('RGB')
    w,h = raw_image.size
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    img_files = glob(args.data + "*")
    img_size = 384

    # load the model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=img_size, vit='base')
    model.eval()
    model = model.to(device)

    captions = dict()

    for img_file in img_files:
        image_file_name = img_file.split('/')[-1]
        image = load_demo_image(img_file, img_size, device)
        with torch.no_grad():
            # please make sure that transformers is a lower version number (4.15 etc.) or the inference can raise ValueError
            caption_content = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            # this inference type can generate more detailed captions.
            #caption_content = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
            captions[image_file_name] = caption_content[0]
    with jsonlines.open(os.path.join(args.data, "metadata.jsonl"), "w") as f_out:
        for key in captions.keys():
            f_out.write({"file_name": key, "text": captions[key]})
    #print(captions)

if __name__ == "__main__":
    main()
