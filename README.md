# **BOSCH's Age and Gender Detection**

The goal of the project is to detect age and gender of peoples from given low quality CCTV surveillance video. For the same purpose, we have broken the problem statement into three parts:


1.   Detecting Persons from the surveillance video and extracting the faces from it.
2.   Enhancing the photo thus obtained through GAN models.
3.   Predicting the gender and age from the super-resoluted image obtained.


The above components have been named as M1, M2 and M3 respectively.

The codes have been written in Google Colabotary notebook.

## **Pre-requisites for running the code:**



*   The colab must be connected to GPU runtime.
*   The version of the libraries should be the following:
    *  numpy : 1.21.5
    *  openCV : 4.1.2
    *  torch : 1.10.0+cu111
    *  matplotlib : 3.2.2
    *  Pillow : 7.1.2
    *  pytube : 12.0.0
    *  gdown : 4.2.2
    * basicsr>=1.3.3.11
    * facexlib>=0.2.0.3
    * gfpgan>=0.2.1



## **How to run the code?**


* First two cells of the ipynb installs the necessary libraries needed for stage 1.
```
!pip install --upgrade --no-cache-dir gdown
```
```
!pip install pytube
```
* The next three cells downloads the required data such as pretrained wights of the models and other necessary data. 
```
!gdown --id '10NaxFCpitXjtLX0rZ4M02FV0rOGRpLKI' #weights_dummy.pt 
```
```
!gdown --id '1rJI17Y2u0MDmqv2qkcM6ScOYZpGE_9dd' #weights.caffemodel
```
```
!gdown --id '1M38YE0S9ZztaKK9JJ1GeBbUqIunAimRV' #deploy.prototext file for caffemodel
```

* **Kindly run these cells individually as it might take few seconds to load the large files into the workspace and also ensure that weights_dummy.pt, weights.caffemodel and deploy.prototext gets imported in the colab workspace.**

After doing so, you can run the rest of the code. The training has been skipped and the model is loaded with pretrained weights. In case, you want to train the stage 1 of the model, then you need to uncomment the cells just below heading "Uncomment the following cells to train the model again, in that case comment out/do not run the 'Training is skipped and model is loaded with pretrained weights' section." and comment out the next section.


> **Note:** It might take some time to clone some repositories in the notebook.

In the get_bbox function there are two parameters involved.
```
num_people = len(scores[scores > 0.7])
```
Here those persons are selected where the confidence is greater than 0.7. This value can be changed as per user's requirement.
```
  indices =   torch.ops.torchvision.nms(torch.Tensor(boxes),torch.Tensor(scores),0.25).tolist()

```
Here those boxes are chosen where the IOU is less than 25%. Again this threshold can be changed as per user's requirement.

In the function ```videoDataProcessing``` we are capturing frames after every 20 milliseconds,
```
while cap.isOpened():
        ret, frame = cap.read()
        if ret and i%20==0:
            get_bbox(frame,model,i/20,k)
            print(i)
        elif not ret:
            break
        i+=1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
```
this value can be changed by replacing some other value in place of 20.

* After running these functions, the program will pause and ask the user for the input. ![input.jpg](https://user-images.githubusercontent.com/75763525/159293859-37e20f18-ac56-4321-93bc-2fd2f12707de.jpg)

  * The user have the flexibility to provide the input either as images or videos. 
  * The user can provide the video input in two formats, i.e., either as a youtube video link or upload the video from the local system. 
  * For providing video inputs, first you need to give 0 as input and again you will be asked for input. Enter 0 if you want to enter youtube video link otherwise it will ask to upload files from the system.
  ![yt.jpg](https://user-images.githubusercontent.com/75763525/159293988-d1210c02-d96a-4918-a3f5-9a9a2406bfb1.jpg)

  * For providing image inputs, the user need to enter 1 and choose the images **only from the local system**. 
  

> **Note:** It might take some time to upload the images/videos into the google colab workspace depending upon the speed of the network being used.

> **Note:** Kindly refrain from using YouTube video link whose content length is more than 5 minutes, as it might not be processed by pytube library. 


Once the inputs are provided the model will automatically detect the persons and crop the person's body as well as the faces and will save in different folders in the colab workspace, namely m1_cropped and faces respectively.

These images thus obtained will be fed to Real ESRGAN (M2) and it will super resolute the images. 

* The super resoluted body images will get saved in the folder body_images in colab workspace.
* The super resoluted face images will get saved in the folder face_images in colab workspace.

