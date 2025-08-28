# Hand Gesture Recognition Datasets

Datasets (Available if required). Corresponding Email: [huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)

* The dataset contains 10 human hand pose from more than 10 volunteers.
* The FMCW Radar used for data acquisition in this project is a mmWave Radar sensor [AWR1243BOOST](https://www.ti.com/tool/AWR1243BOOST) and [DCA1000EVM](https://www.ti.com/tool/DCA1000EVM), both manufactured by Texas Instruments (TI). 

# Dataset structure should look like the ImageNet
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
