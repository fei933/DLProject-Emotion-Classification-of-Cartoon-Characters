# Detecting Cartoon Characters’s Emotions Using Transfer Learning

Feifei Wang, Yutong Zhang

## Goal

Emotion Classification of Cartoon Characters of different style (anime vs. 3D cartoon)

## **Solution**

- Transfer learning and fine-tuning
- Compare the accuracy of different pretrained models & baseline CNNs

## Values

- Few studies explored this subject before => improve image search result quality
- Understand how neural networks differentiate emotions of fictional figures, whose characteristics vary dramatically between artists

## Challenges

- Animated faces have different characteristics from real human faces
- The cartoon facial emotion datasets are limited, with small sizes that is susceptible to overfitting

## Data Sets

1. [Facial Expression Research Group 2D Database (FERG-DB)](http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html)

55767 annotated face images of 6 characters

```bash
{'angry': 0,
  'crying': 1,
  'embarrassed': 2,
  'happy': 3,
  'pleased': 4,
  'sad': 5,
  'shock': 6 }
```

2. **[Manga Facial Expressions Data Set (462 images)](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions)**

```bash
{'anger': 0,
 'disgust': 1,
 'fear': 2,
 'joy': 3,
 'neutral': 4,
 'sadness': 5,
 'surprise': 6 }
```

## Code Structure

all in .ipynb, separated by models and datasets

- trained-from-scratch CNN （作为baseline model，其他的accuracy可以和它compare）
- GoogleNet
- ResNet50

## Result

### GoogleNet
![](https://github.com/fei933/DLProject-Emotion-Classification-of-Cartoon-Characters/blob/main/images/googlenet1.png)
![](https://github.com/fei933/DLProject-Emotion-Classification-of-Cartoon-Characters/blob/main/images/googlenet2.png)
![](https://github.com/fei933/DLProject-Emotion-Classification-of-Cartoon-Characters/blob/main/images/googlenet3.png)

- Use `’val_categorical_accuracy’` to evaluate accuracy
- Overall top 3 performance:
    - L2 Regularization
    - baseline + Batch Norm + 2 Dense 64 Layer
    - baseline
- When the dataset is small
    - Changing the structure of the model is able to increase the accuracy and control overfitting, with mild effect on runtime
- When the dataset is large:
    - GoogleNet is faster the baseline

### ResNet50 and Vanilla CNNs
![](https://github.com/fei933/DLProject-Emotion-Classification-of-Cartoon-Characters/blob/main/images/r****resnet-vanilla-manga.png)
![](https://github.com/fei933/DLProject-Emotion-Classification-of-Cartoon-Characters/blob/main/images/resnet-vanilla-ferg.png)
