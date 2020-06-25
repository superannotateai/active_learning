# Active Learning algorithm for image sorting before annotation

Active Learning algorithms allow users to select a subset of his/her not yet annotated images in such a way, that annotating and adding them to the training dataset will result in highest improvement in the model's accuracy. Annotate.online allows users to upload 'entropy' csv files, which which contain information on the 'entropy' value for each image, and annotate the images in the decreasing order of entropies. This way users with limited budget can prioritize images to be annotated.


## Code Structure

Here we provide an implementation of the active learning algorithm [Learning Loss for Active Learning"](https://arxiv.org/pdf/1905.03677.pdf) for classification, object detection and segmentation. Code for each application is provided in the coorresponding folder. Our code runs for a few 'cycles' and selects 1000 images per cycle, then trains the model on those 1000 images. This way we mimic future user's actions, as generation of entropy values for all the images at once and annotating top N images will result in worse performance compared to repeating the process for a few cycles and re-training the model after each cycle. The code generates csv files for each cycle, which can be uploaded to annotate.online.

## Used open-source model codes

For each task we integrated our active learning code into an open-source repository to demonstrate it's usage. Please refer to the corresponding repo's instructions for setup.

Classification on Cifar 10 - https://github.com/kuangliu/pytorch-cifar
Object Detection on Paskal VOC using SSD algorithm - https://github.com/amdegroot/ssd.pytorch
Segmentation on cityscapes with DRN algorithm - https://github.com/fyu/drn

## How to add active learning to my module?

In order to add active learning to your model, the following steps must be performed:

1. Copy the 'active_learning' folder to your code.
2. Implement functions 'get_active_learning_feature_channel_counts' and 'get_active_learning_features' inside your module. They will provide features to the active learning loss prediction module. Take a look into our sample code for the reference. 
3. Add active learning loss prediction module to your model with the following lines:
```
from active_learning import ActiveLearning
from active_loss import LossPredictionLoss
from active_learning_utils import choose_active_learning_indices, random_indices

....
# This adds a few layers to your initial model.
net = ActiveLearning(net)
....

# Change outputs = net(inputs) in your training function to
outputs, loss_pred = net(inputs)
....
# Add the active learning loss to your model loss like this:
criterion_lp = LossPredictionLoss()
lp = lamda * criterion_lp(loss_pred, loss)
loss += lp
```

4. Use function 'choose_active_learning_indices' to select images to annotate next. It will return image indexes in the dataset and corresponding entropy values. Use function 'write_entropies_csv' for creating the csv file, or create a similiar function.
 

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.
