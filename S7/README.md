# S7 Assignment:

Your new target is:

* 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 8000 Parameters
* Do this using your modular code. Every model that you make must be there in the model.py file as Model\_1, Model\_2, etc.

## Experimentation :

Note : 
* [model.py](model.py) has model_1,model_2,model_3,model_4 classes
* [utils.py](utils.py) has train/test scripts
* Notebooks : Model_V1/V2/V3/V3a/V4 uses modular code from model.py and utils.py files.
       

### **Model : Model\_1**

Targets:
* Consistent 99.4% Test accuracy , Less than 8k Parameters

* Model : Model\_1
* Notebook : [Model\_V1.ipynb](Model_V1.ipynb)
* Results:
    * Training Accuracy : 99.96
    * Test Accuracy : 99.51
    * Parameters : 104,762

### Analysis:

* Model Overfits after 7 epoch.
* Scope to reduce parameters and add dropout layer to regulate overfit.
* Model reached best test accuracy at 10th epoch.

**Model\_V1 Summary:**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
            Conv2d-3           [-1, 64, 24, 24]           9,216
       BatchNorm2d-4           [-1, 64, 24, 24]             128
            Conv2d-5           [-1, 32, 24, 24]           2,048
         MaxPool2d-6           [-1, 32, 12, 12]               0
            Conv2d-7           [-1, 64, 10, 10]          18,432
       BatchNorm2d-8           [-1, 64, 10, 10]             128
            Conv2d-9             [-1, 64, 8, 8]          36,864
      BatchNorm2d-10             [-1, 64, 8, 8]             128
           Conv2d-11             [-1, 64, 6, 6]          36,864
      BatchNorm2d-12             [-1, 64, 6, 6]             128
        AvgPool2d-13             [-1, 64, 1, 1]               0
           Conv2d-14             [-1, 10, 1, 1]             640
================================================================
Total params: 104,752
Trainable params: 104,752
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.10
Params size (MB): 0.40
Estimated Total Size (MB): 1.50
----------------------------------------------------------------
```

### **Model Version 2 : Model\_2**

Targets:
* Consistent 99.4% Test accuracy , Less than 8k Parameters
* Build lighter Model with decent accuracy
* Model : Model\_2
* Notebook : [Model\_V2.ipynb](Model_V2.ipynb)
* Results:
  * Training Accuracy : 99\.56
  * Test Accuracy : 99\.33
  * Parameters : 9873

### Analysis:

* Reduced parameter < 10k by playing around with reducing size of channels
* Model seems to be overfitting
* Model doesnot reach to 99.4 accuracy

**Model\_V2 Summary:**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 9, 26, 26]              81
       BatchNorm2d-2            [-1, 9, 26, 26]              18
            Conv2d-3           [-1, 18, 24, 24]           1,458
       BatchNorm2d-4           [-1, 18, 24, 24]              36
            Conv2d-5           [-1, 12, 24, 24]             216
         MaxPool2d-6           [-1, 12, 12, 12]               0
            Conv2d-7           [-1, 18, 10, 10]           1,944
       BatchNorm2d-8           [-1, 18, 10, 10]              36
            Conv2d-9             [-1, 18, 8, 8]           2,916
      BatchNorm2d-10             [-1, 18, 8, 8]              36
           Conv2d-11             [-1, 18, 6, 6]           2,916
      BatchNorm2d-12             [-1, 18, 6, 6]              36
        AvgPool2d-13             [-1, 18, 1, 1]               0
           Conv2d-14             [-1, 10, 1, 1]             180
================================================================
Total params: 9,873
Trainable params: 9,873
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.37
Params size (MB): 0.04
Estimated Total Size (MB): 0.41
----------------------------------------------------------------
```

### **Model Version 3: Model\_3**

Targets:
* Consistent 99.4% Test accuracy , Less than 8k Parameters
* Reduce Overfitting and Improving Accuracy by applying RandomRotation
* Model : Model\_3
* Notebook : [Model\_V3.ipynb](Model_V3.ipynb)
* Results:
    * Training Accuracy : 98.39
    * Test Accuracy : 99.36
    * Parameters : 9873

### Analysis:

* Added Dropout layer to Handle Overfitting
* Added Transformation (RandomRotation) augmentation
* It is evident from Train and Test accuracy that Model is generalizing well not Overfitting
* Observed that after 8th Epoch Model may stuck at local Minima. Probably LR schedular to be used to get improve learning.
Targets:
* Improve Learning using LR Schedular.
* Model : Model\_3
* Notebook : [Model\_V3a.ipynb](Model_V3a.ipynb)
* Targets: Handle Overfitting and Improve Accuracy
* Results:
    * Training Accuracy : 99.42
    * Test Accuracy : 99.43
    * Parameters : 9873

### Analysis:

* Added Dropout layer to Handle Overfitting with Dropout value 0.01
* Added Transformation (RandomRotation) augmentation
* Model is not Overfitting and Generalizing well with LR scheduler at step 10
* From Epoch 11 - Model Consistently giving test accuracy > 99.4.
* Next step is to reduce parameter to make it < 8k

**Model\_V3 Summary:**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 9, 26, 26]              81
       BatchNorm2d-2            [-1, 9, 26, 26]              18
         Dropout2d-3            [-1, 9, 26, 26]               0
            Conv2d-4           [-1, 18, 24, 24]           1,458
       BatchNorm2d-5           [-1, 18, 24, 24]              36
         Dropout2d-6           [-1, 18, 24, 24]               0
            Conv2d-7           [-1, 12, 24, 24]             216
         MaxPool2d-8           [-1, 12, 12, 12]               0
            Conv2d-9           [-1, 18, 10, 10]           1,944
      BatchNorm2d-10           [-1, 18, 10, 10]              36
        Dropout2d-11           [-1, 18, 10, 10]               0
           Conv2d-12             [-1, 18, 8, 8]           2,916
      BatchNorm2d-13             [-1, 18, 8, 8]              36
        Dropout2d-14             [-1, 18, 8, 8]               0
           Conv2d-15             [-1, 18, 6, 6]           2,916
      BatchNorm2d-16             [-1, 18, 6, 6]              36
        Dropout2d-17             [-1, 18, 6, 6]               0
        AvgPool2d-18             [-1, 18, 1, 1]               0
           Conv2d-19             [-1, 10, 1, 1]             180
================================================================
Total params: 9,873
Trainable params: 9,873
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.53
Params size (MB): 0.04
Estimated Total Size (MB): 0.57
----------------------------------------------------------------
```

### **Model Version 4: Model\_4**

Targets:
* Consistent 99.4% Test accuracy , Less than 8k Parameters
* Model : Model\_4
* Notebook : [Model\_V4.ipynb](Model_V4.ipynb)

## Results :

* Model : Model\_4
* Targets: 99.4% Accuracy and < 8k Parameter Achieved
* Results:
    * Training Accuracy : 99.55
    * Test Accuracy : 99.42
    * Parameters : 7901

### Analysis:

* Model consistently achieved 99.4 Accuracy in 9 and 10 Epoch.

**Model\_V4 Summary:**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 7, 26, 26]              63
       BatchNorm2d-2            [-1, 7, 26, 26]              14
         Dropout2d-3            [-1, 7, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           1,008
       BatchNorm2d-5           [-1, 16, 24, 24]              32
         Dropout2d-6           [-1, 16, 24, 24]               0
            Conv2d-7           [-1, 12, 24, 24]             192
         MaxPool2d-8           [-1, 12, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,728
      BatchNorm2d-10           [-1, 16, 10, 10]              32
        Dropout2d-11           [-1, 16, 10, 10]               0
           Conv2d-12             [-1, 16, 8, 8]           2,304
      BatchNorm2d-13             [-1, 16, 8, 8]              32
        Dropout2d-14             [-1, 16, 8, 8]               0
           Conv2d-15             [-1, 16, 6, 6]           2,304
      BatchNorm2d-16             [-1, 16, 6, 6]              32
        Dropout2d-17             [-1, 16, 6, 6]               0
        AvgPool2d-18             [-1, 16, 1, 1]               0
           Conv2d-19             [-1, 10, 1, 1]             160
================================================================
Total params: 7,901
Trainable params: 7,901
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.46
Params size (MB): 0.03
Estimated Total Size (MB): 0.49
----------------------------------------------------------------
```
