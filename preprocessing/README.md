<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

We follow the preprocessing code used in [edm2](https://github.com/NVlabs/edm2). In this code we made a several edits: (1) we removed unncessary parts except preprocessing because this code is only used for preprocessing, (2) we use [-1, 1] range for an input to the stable diffusion VAE (similar to DiT or SiT) unlike edm2 that uses [0, 1] range, and (3) we consider preprocessing to 256x256 resolution (or 512x512 resolution).

To download, go to https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data, signup, accept the rules, find the Download All button, open inspect element and go to network tab, click download all button, find the googleapis link to the archive.zip. Something like 

```
https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1751647197&Signature=dyPSKcJlq5YFL3eDb791xiKS0FcUXcdZ9xaZ0Kvu6SNE8mp8OqZM938RKEVfP6C%2FE5i9QcQbGo1UZhEEp1Q5UYChKDZEWPg26zytPnwmyX2biytjzW215p3EKLFwECpsUFUb1OAUG8xJoU0bUtRyN82wrhFvh2BOIYbcRlDImIzmd%2B7oA69b9JmuOcnpJiy4S9IS31usvK5eJ%2BhpFDtabMwW%2B9kqc8LIIdLFpJyi0qMnqfEM8ZetvDNxrOvocigMsfCdMB8a0iJC%2F%2BClP%2BpHyypgIfHdnqTwV0sudc6Ibh0X4zUfOTwVMrTYiPU0ZWBzlDJsjcRTourOBNvM9W5%2BpQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dimagenet-object-localization-challenge.zip
```
(This link will be expired by the time you read it, so make a new link)

then use ```aria2c -x 16 -s 16 "link"``` to download quickly

After downloading ImageNet, please run the following scripts (please update 256x256 to 512x512 if you want to do experiments on 512x512 resolution);

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=data/ILSVRC/Data/CLS-LOC/train \
    --dest=dataset/images --resolution=256x256 --transform=center-crop-dhariwal
```

```bash
# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=dataset/images \
    --dest=dataset/vae-in
```

## Acknowledgement

This code is mainly built upon [edm2](https://github.com/NVlabs/edm2) repository.