# Organic and Anorganic waste detection 

[This dataset](https://universe.roboflow.com/mallail-qadrillah-m353dsx0160/klasifikasi-sampah-0ydou) was exported via roboflow.com on October 17, 2023 at 2:49 PM GMT

The dataset includes 931 images with 8 classes.
1. Ampas tebu
2. Ranting Kayu
3. aqua botol
4. aqua gelas
5. bungkus kopi sachet
6. daun
7. kulit telur
8. plastik minyak goreng

Ampas-tebu-Ranting-Kayu-aqua-botol-aqua-gelas-bungkus-kopi-sachet-daun-kulit-telur-plastik-minyak-goreng are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -27 and +27 percent
* Random Gaussian blur of between 0 and 1.5 pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random brigthness adjustment of between -25 and +25 percent

dataset trained with pretrained model **ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8**

### Result of the detection
![deteksi_kulit_telur](https://github.com/allail-qadrillah/Organic-and-Anorganic-waste-detection/assets/89723505/89e53342-aa95-4a20-8112-a7099f40ef70)
![deteksi_plastik_minuman_gelas](https://github.com/allail-qadrillah/Organic-and-Anorganic-waste-detection/assets/89723505/526afd3d-e723-4e7c-a015-04285cf82cf1)

You can read all about this project at [this link](https://octagonal-pressure-9f0.notion.site/Sampah-Image-Detection-3920a973351b4d65aef7b0b49e79de9d?pvs=4)
