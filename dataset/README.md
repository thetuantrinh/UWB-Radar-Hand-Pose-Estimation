# Hand Pose Estimation Datasets

Datasets (Available if required). Corresponding Email: [huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)

* The dataset contains 10 human hand pose from more than 10 volunteers.
* The FMCW Radar used for data acquisition in this project is a mmWave Radar sensor [AWR1243BOOST](https://www.ti.com/tool/AWR1243BOOST) and [DCA1000EVM](https://www.ti.com/tool/DCA1000EVM), both manufactured by Texas Instruments (TI).
* The camera used for teacher’s ground-truth in this project is [Logitech C270 HD Webcam](https://www.amazon.com/Logitech-C270-HD-Webcam-Black/dp/B008QS9J6Y/ref=sr_1_9?dib=eyJ2IjoiMSJ9.xVtRFzFOfA678C9UfJ2P5AXLMD7G6OTRv-G1dBoLjUMTVBRP9yMqLRNl-wd4oocZR4DnBkvqqYbEJRCFtetROQ8HsI8oFaaQp2IqjPM-3L1PwieISIRhEevy5tb2enV6ZpfmYr7XxcE192Dtq-YgKJGVGbRVIZ8EkjS-bomtGFueqbvyocCV0enew3wVZgm1fhtJoOgBwIlVimjo5Ubn87DuHjQtlibJ_Iw5ygMwzP0.ZX78OwSfjzXmXtZrHc9M68wdYt8yVTGWJOebUkfPvVk&dib_tag=se&keywords=logitech+camera&qid=1756371132&sr=8-9)

## Dataset structure should look like the ImageNet
```bash
├── dataset
│   ├── images
│   │   ├── empty
│   │   │   └── 0.png
│   │   ├── left
│   │   │   └── 0.png
│   │   └── right
│   └── radar
│       ├── empty
│       │   └── 0.npy
│       ├── left
│       │   └── 0.npy
│       └── right
└── README.md

9 directories, 5 files
dataset/
├── images
│   ├── empty
│   │   └── 0.png
│   ├── left
│   │   └── 0.png
│   └── right
├── posed_img
│   └── run1
│       └── 0.png
├── radar
│   ├── empty
│   │   └── 0.npy
│   ├── left
│   │   └── 0.npy
│   └── right
└── undetected
    ├── images
    │   └── 0.png
    └── radar
        └── 0.npy

13 directories, 7 files
```
