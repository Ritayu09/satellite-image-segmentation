### Project Description:

1km x 1km satellite images in both 3-band and 16-band formats are provided and the goal is to detect and classify the types of objects found in these regions. 16-band images are furthur provided into 2 different 8-band image files. These bands are denoted as A, M and P bands.   

[Data Link](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)

In the project, we will utilize the images from the M band (8 channel image files) to build an image segmentation model to classify and detect 10 classes in the satellite images. These 10 classes are as following:

1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
2. Misc. Manmade structures 
3. Road 
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway 
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike

The label information is provided in WKT format and we will have to generate image mask for all training data using this information.

The modelling framework in the project involves UNET architecture wherein we first downsample the data to learn features from the images and then upsample to generate the mask for the input image. Convolution and pooling layers are used to downsample and transpose convolution layers were used to upsample data. Advantage of using transpose convolution layers over upsampling layers is that these layers are trainable (performs inverse convolution operation) whereas upsample layers simply double the dimension by interpolating pixel values making them non-trainable layers.

### Python Libraries Utilized in Project:

- os, shuntil: High level operations on files and folders in directory
- tiff: for loading image data into python
- Pandas, Numpy: For image data preperation, manipulation and cleaning
- shapely: To handle image label data provided in WKT format
- MatplotLib: For image visualization and plotting model results
- cv2: OpenCV library for image data manipulation and image mask generation
- Keras, Tensorflow: For building convolution neural network and training model
