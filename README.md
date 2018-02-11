本工程旨在演示如何使用imgaug建立较完整的数据增强流程。
在garylau/imgaug.py中使用的数据标定的存储的xml格式和PASCAL VOC的xml格式是相同的。
使用https://github.com/GarryLau/labelImg 进行数据标定，其xml格式就是和PASCAL VOC数据标定的xml格式相同。
train-GroundTruth.txt中内容格式详见https://github.com/GarryLau/imgaug/blob/master/garylau/train-GroundTruth.txt 。

在实际应用的时候建议一次对图像进行变换时采用1~3个变换，这样容易看出变换的效果是否合乎预期。另外，变换的方式越多会导致变换出现异常的概率增大，比如在做检测任务时如果对图像进行增强，进行Affine和CropAndPad的联合操作时很容易将感兴趣区域（目标）移出图片，这样的图片就毫无意义，如果加入训练的话会导致loss出现nan值。
