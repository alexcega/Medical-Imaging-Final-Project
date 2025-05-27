# MI Project: 3D DICOM data

This repository contains code for loading, processing, and visualizing the RadCTTACEomics CT and segmentation data. It includes steps to load the DICOM images, reorder slices, extract and align tumor masks, and generate a rotating Maximum Intensity Projection (MIP) animation with the tumor overlay.
And the registration between 2 ct files
## Repository Structure

```
MI-Project/
├── 1196/                   # Raw DICOM data (not in repo)
├── results/
│   └── MIP/                # Generated projection images and animations
├── point1.ipynb
├── point2.ipynb
├── README.md               # Project overview and instructions
└── .gitignore              # Ignored files and folders
```


## Usage

1. **Place** your dataset under `data/` (e.g., `data/1196/30_EQP_Ax5.00mm` and `data/1196/ManualROI_Tumor.dcm`).
2. **Run the processing script notebook**:
DICOM loading and visualization  
   ```
   point1.ipynb
   ```
   3D Rigid Coregistration
   ```
   point2.ipynb
   ```
3. **View** the generated animation at `results/MIP/rotating_MIP.gif`.
![mask](RadCTTACEomics_1196\results\MIP\rotating_liver_tumor.gif)
