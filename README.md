# Self-Supervised-Multimodal-Medical-Image-Analysis
## Description
Use state-of-the-art self-supervised deep learning methods such as DINO (DINO: Emerging Properties in Self-Supervised Vision Transformers), Image-GPT, and BYOL (Bootstrap your own latent: A new approach to self-supervised Learning) to investigate the performance of these algorithms on medical data. The project aims to use multiple modalities of medical images, from MRI's to CT's and MG's of various parts of the human body: brain, chest, knee, breast, etc. We want to develop training techniques and models that, after being trained on a large unlabeled collection of datasets, can outperform models trained from scratch on downstream tasks. 

## Setup
1. Download the data from Kaggle: [link](https://www.kaggle.com/c/siim-covid19-detection/data). I recommend downloading the data using their command line utility (its a python package)
2. Run the file structure cleanup [script](misc/dataset_folder_cleanup.py), once in the train folder and once in the test folder. 
3. Set the path to your data directory in the enviorment variables by running this command:
```Command
SETX TENSOR_RELOADED_DATADIR "E:\Data"
``` 
Where you replace E:\Data with the path to where you store your datasets. See an example file structure bellow ![Example data structure](misc/Example-data-folder-file-structure.png)

4. Copy all the csv files found in the 'misc' folder to your dataset root folder (where the train and test directories are found)
5. Install the python packages from the requirements file (WIP)
```Command
pip install -r requirements.txt
```

## Datasets:

### 1. [fastMRI Dataset](https://fastmri.med.nyu.edu/)
- Description: unlabeled knee and brain MRI scans (3D)
- Size: 1500 knee scans and 7000 brain scans
- Request for data: Approved (Contact Cristi for download links)
- Included data tasks/labels: TODO

### 2. [National Lung Screening Trial](https://cdas.cancer.gov/projects/nlst/796/overview/)
- Description: lung cancer CT's (3D)
- Size: 26254 subjects
- Request for data: Approved (Contact Cristi for download links)
- Included data tasks/labels: TODO

### 3. [The VICTRE Trial: Open-Source, In-Silico Clinical Trial For Evaluating Digital Breast Tomosynthesis](https://wiki.cancerimagingarchive.net/display/Public/The+VICTRE+Trial%3A+Open-Source%2C+In-Silico+Clinical+Trial+For+Evaluating+Digital+Breast+Tomosynthesis)
- Description: breast cancer MG scans (2D X-ray projection and 3D volume I think*)
- Size: 2994 subjects
- Request for data: Public data
- Included data tasks/labels: TODO

### 4. [The Chinese Mammography Database (CMMD)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508#70230508bcab02c187174a288dbcbf95d26179e8)
- Description: breast cancer MG scans (3D)
- Size: 1775 subjects
- Request for data: Public data
- Included data tasks/labels: TODO

### 5. [Prostate MRI and Ultrasound With Pathology and Coordinates of Tracked Biopsy (Prostate-MRI-US-Biopsy)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661)
- Description: Prostate Cancer MRI and US (ultra-sound) (3D)
- Size: 1151
- Request for data: Public data
- Included data tasks/labels: TODO

### 6. [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- Description: Lung Cancer CT, CR, DX (3D)
- Size: 1010
- Request for data: Public data
- Included data tasks/labels: TODO

### 7. [CT Images in COVID-19](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19)
- Description: COVID-19 Chest CT (3D)
- Size: 632
- Request for data: Public data
- Included data tasks/labels: TODO

### 8. [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)
- Description: Brain CT scans (3D)
- Size: 700k ???
- Request for data: Kaggle competition rule acceptance (Done)
- Included data tasks/labels: TODO

### 9. [MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/)
- Description: Knee MRI's
- Size: 1370 subjects
- Request for data: Approved (Contact Cristi for download links)
- Included data tasks/labels: TODO

### 10. [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview)
- Description: Chest X-Ray's (2D)
- Size: 6334
- Request for data: Kaggle competition rule acceptance
- Included data tasks/labels: object detection and classification of Pneumonia

### 11. [BIMCV-COVID19](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711)
- Description: Chest X-Ray's (2D)
- Size: 7377 CR, 9463 DX
- Request for data: [Public data](https://b2drop.bsc.es/index.php/s/BIMCV-COVID19-cIter_1_2)
- Included data tasks/labels: object detection and classification of Pneumonia

### 12. [Chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- Description: Chest X-Ray's (2D)
- Size: 224,316 chest radiographs of 65,240 patients
- Request for data: Public data (Contact Cristi for download links)
- Included data tasks/labels: ???

### 13. [MIMIC](https://mimic.mit.edu/docs/gettingstarted/)
- Seems too have multiple versions, need to investigate


## Introductory concepts
### Transformers/Attention mechanisms: 
https://www.youtube.com/watch?v=iDulhoQ2pro
### Visual transformers: 
https://www.youtube.com/watch?v=TrdevFK_am4
Idea: Instead of spliting input into patches, simulate a convolution looking at the data. Meaning, instead of havving a 16 stride when making 16x16 patches, have smaller strides which will make the "patches" to overlap. Even more adjusting to patch creation can be done with padding, dilution, strides and transposing
### Image-GPT: 
https://www.youtube.com/watch?v=YBlNQK0Ao6g
### ResNets: 
https://www.youtube.com/watch?v=GWt6Fu05voI

## Advanced concepts
### Attention approximation:
- Performers: https://www.youtube.com/watch?v=xJrKIPwVwGM
- Hopfield Layers: https://www.youtube.com/watch?v=nv6oFDp6rNQ
- Switch Transformers: https://www.youtube.com/watch?v=iAR8LkkMMIM
- Nyströmformer: https://www.youtube.com/watch?v=m-zrcmRd7E4
### Adaptive Gradient Cliping: 
https://www.youtube.com/watch?v=rNkHjZtH0RQ
### Disentangled Attention: 
https://www.youtube.com/watch?v=_c6A33Fg5Ns 
Idea: use this for using relative positioning but also try to compute more attention matrices. Not only C-P, P-CP and C-C but also use normal attentions into it like CP-C, CP-CP, C-CP, P-CP)
### Transformer in Transformer:
https://arxiv.org/abs/2103.00112
Idea: Do TNTNT for 3+ levels of details for the transformer, combined with the sliding window patching and experiment with not adding the pixel embedding to the normal patch embedding, and just use the pixel embeddings, or try not to pass the pixel tokens thought the linear layer, just use them raw.

### Self-supervised stuff:
- https://www.youtube.com/watch?v=YPfUiOMYOEE
- https://www.youtube.com/watch?v=h3ij3F3cPIk
- https://www.youtube.com/watch?v=Elxn8rS88bI

Maybe using augmentations such as this one for the self-supervised task: [Attentive CutMix](https://arxiv.org/abs/2003.13048)

### Transfomer stuff:
https://www.youtube.com/watch?v=2PYLNHqxd5A


### To investigate: 
- [ ] https://radrounds.com/radiology-news/list-of-open-access-medical-imaging-datasets/
- [ ] https://www.aylward.org/notes/open-access-medical-image-repositories
- [ ] https://github.com/sfikas/medical-imaging-datasets
- [ ] https://www.cancerimagingarchive.net/access-data/

### Cool way to present results:
http://demos.md.ai/#/bone-age
