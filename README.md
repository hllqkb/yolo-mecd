# YOLO-MECD
1.Added EMA-attention-module and EMA yaml

2.Choose the CSPPC module, designed based on partial convolution, replaces the C3K2 module

3.Adopted the MPDIoU loss function instead of the CIoU loss function, which not only improves detection accuracy but also accelerates convergence speed.

4.Improved Conv convolution with ADown module in YOLOV9,based on YOLOV11 lightweight downsampling operation to reduce parameter count