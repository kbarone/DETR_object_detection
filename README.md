# Object Detection with DETR 

This is an end-to-end objection detection model using Facebook’s Detection Transformer (DETR) model. The dataset I used was the binary classification Global Wheat Detection dataset.

First, we needed to create new dataloaders for the wheat dataset. I read in the train.csv document and broke out the bounding box column into four separate columns. 

![image](https://user-images.githubusercontent.com/89941817/225946828-8bd45be0-86c1-4ba5-94e6-8497c394d50f.png)
 
I also chose to break out the data into stratified folds so we could visualize a model for each fold. I created a new dataframe that included image ids and the group and stratification group which sample was associated with which K fold.

![image](https://user-images.githubusercontent.com/89941817/225946968-35dccbd5-f730-4e70-b6a4-a686d2d9a8c4.png)

 
I created a class called WheatData that takes in image IDs, a dataframe, and transforms to be applied to the images. WheatData then normalizes the images and their associated bounding boxes and transforms relevant objects into tensors. Using this WheatData class, we can create datasets for training and validation data, and feed them into DataLoaders with a batch size of 8, and 4 workers.

From the Facebook repo, I downloaded their loss tracking function (HungarianMatcher()) and outlined the different losses we need to track (classification loss for labels, bounding box loss and loss for the non-wheat class). My actual model class is fairly simple as all the DETR model specifications are outlined in the Facebook repo. I load in the pretrained “detr_resnet50” model and indicate that we only want to output two classes in our final linear layer. I also create the model using 100 queries to indicate we would accept up to 100 bounding boxes that identify wheat in an image.

The training and evaluation functions are similar to those we’ve worked with in the past, the main difference being that we are now accounting for potentially three different loss types. I also chose to use an Adam optimizer for this model. The “run” function trains a model using four of five folds of data and uses the last fold as a validation set.

The output of the run for each of the five folds is below. I’ve included both training and validation loss after each epoch, along with a representative image with predicted and true boxes after three epochs. The boxes and images are visualized using the “view_boxes” function. I chose to run only three epochs after seeing that the predictive power was quite high after only a few epochs, and the models were taking a good bit of time to run, so to save some time, I reduced the number of epochs from five to three.

Fold 1:

![image](https://user-images.githubusercontent.com/89941817/225947030-8ef7cd7f-1353-4905-af25-7c23916c3344.png)


![image](https://user-images.githubusercontent.com/89941817/225947077-dd2adb01-6b3f-4a96-bbd4-6bef902f1d1a.png)

 

Fold 2:
 
![image](https://user-images.githubusercontent.com/89941817/225947127-17e02071-3eea-4fd3-b63c-a90b9d7a4d41.png)

![image](https://user-images.githubusercontent.com/89941817/225947148-87681103-609e-4a10-b86f-d8faedc28cf8.png)

 
Fold 3
 
  ![image](https://user-images.githubusercontent.com/89941817/225947184-57d51afc-b6e8-4ad4-8da5-3df5e4939f9d.png)

![image](https://user-images.githubusercontent.com/89941817/225947217-895eed02-c5fc-4df9-b5d2-d35cda442c90.png)




Fold 4:
 
![image](https://user-images.githubusercontent.com/89941817/225947269-22afbdaa-baf5-41e1-8f40-4c37062e2e72.png)

![image](https://user-images.githubusercontent.com/89941817/225947287-a3fbdf68-acf1-49b6-b912-d234d0806242.png)




Fold 5:

![image](https://user-images.githubusercontent.com/89941817/225947334-f1aa2635-5225-4565-a711-046356f0c89e.png)

![image](https://user-images.githubusercontent.com/89941817/225947373-1f3a341d-3856-42f7-a788-fe86e6126065.png)
