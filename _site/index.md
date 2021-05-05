Ever wondered how long it will take to create a custom text classifier? This tutorial will demonstrate the process of easily creating a custom text classifier using Flair to predict product review ratings. This tutorial will highlight the following steps:
- Loading a dataset
- Training a classifer
- Predicting a label 

# What is Flair?
[Flair](https://github.com/flairNLP/flair) is an NLP library that supports named entity recognition, part-of-speech tagging, sense disambigusation and classification. Developed by Zalando Research, Flair supports a variety of natural languages out-of-the-box including English, Spanish, German and Dutch. Built directly on PyTorch, the Flair framework provides quick access to training models and trying out custom embeddings and labels.

# Prerequisites
1. Python 3.6+
2. Flair

# Installing Flair
To install Flair to your python project, execute the following pip command:
```
pip install flair
```
Easy right?!

# Library Imports
The tutorial will use the following set of imports from the Flair library:
```python
from flair.data import Sentence, Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings, FlairEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
```

# Loading a Corpus
For this tutorial, we are using a downsized sample of the [_Amazon Reviews for Sentiment Analysis_](https://www.kaggle.com/bittlingmayer/amazonreviews) dataset available on Kaggle. This dataset features over a million Amazon reviews that have been parsed to fastText format. The dataset contains tagged reviews representing two labels: 1 and 2.

   - Label 1: Corresponds to 1- and 2-star reviews
   - Label 2: Corresponds to 4- and 5-star reviews 

_Reviews that had a 3-star rating were considered neutral and discarded from dataset_
Training data examples:
```
__label__1 sucks: overrated fan. just go to lowes or home depot and save money and get something you can return in person.
__label__2 excellent Blue Ray movie: This 5 disk set for $27 is a good deal. 3 blue ray disks and two DVD disks are in the box.
```
Flair also supports [loading datasets in csv format](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md#load-from-simple-csv-file).

The original dataset was downsized into three seperate files:

   - train.txt: Reduced to 8000 lines.
   - dev.txt: Reduced to 1000 lines.
   - test.txt: Reduced to 1000 lines.

To create the corpus, we will use the Flair ```ClassificationCorpus``` class to load our datasets:
```python
# Path to data files
data_path = '../data'
# Load corpus (If datasets are names like below, the test_file, dev_file 
# and train_file fields do not need to be specfied and can be removed from line below.)
corpus = ClassificationCorpus(data_path, 
                              test_file='test.txt', 
                              dev_file='dev.txt', 
                              train_file='train.txt')  
```
After loading the corpus, we will need to initialize the word and document embeddings. Flair offer many embeddings including GloVe, and contextualized [Flair embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md). 

We will be initializing three word embeddings and one document embedding for our classifier:

```python
word_embeddings = [
                    WordEmbeddings('glove'), 
                    FlairEmbeddings('news-forward-fast'), 
                    FlairEmbeddings('news-backward-fast')
                  ]
doc_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)
```

# Training a Classifier
Once our corpus and embeddings are created, we can quickly initialize and train our text classifier. The ```TextClassifier``` class will take in the documented embeddings and created corpus as parameters and use the corpus as the label dictionary. We are now ready to train! 
```python
classifier = TextClassifier(doc_embeddings, label_dictionary=corpus.make_label_dictionary())
# Initialize the model trainer
trainer = ModelTrainer(classifier, corpus)
# Begin training
trainer.train('../data', max_epochs=1)
```
Once training is complete, Flair will have generated several files containing data used, and classifier statistics. It will have created a ```best-model.pt``` dataset that is used to rate the performance of our classifier.
```python
Results:
- F-score (micro) 0.639
- F-score (macro) 0.6301
- Accuracy 0.639

By class:
              precision    recall  f1-score   support

           2     0.6362    0.7476    0.6874       531
           1     0.6436    0.5160    0.5728       469

   micro avg     0.6390    0.6390    0.6390      1000
   macro avg     0.6399    0.6318    0.6301      1000
weighted avg     0.6397    0.6390    0.6337      1000
 samples avg     0.6390    0.6390    0.6390      1000
``` 
Using the configurations in this tutorial, the classifier acheived an F1 score of **0.639**. Due to system constraints, it was not feasible to perform the 150 epochs Flair recommends to achieve _state-of-the-art_ accuracy.

It will also generate in a ```final-model.pt``` file that will be loaded into our ```TextClassifier``` so we can begin predicting labels!
```python
TextClassifier.load('../data/final-model.pt')
```
# Using our Classifier
Now that we have loaded and trained our classifier, we can now prompt users to send the classifier one or more sentences and get a prediction for whether the sentence is a label 1 or 2.
```python
# Predict tags
classifier.predict(sentence)
print(sentence.labels)
input_sentence = input('Enter another sentence to parse or exit: ')
if input_sentence != 'exit':
    predict(Sentence(input_sentence), classifier)
else:
    exit
```
We get the following example output:
```
Enter another sentence to parse or exit: Terrible and disgusting.
[1 (0.5097)]
Enter another sentence to parse or exit: Beautiful shoes!
[2 (0.5593)]
Enter another sentence to parse or exit: Yikes.
[1 (0.52)]
Enter another sentence to parse or exit: Dont spend your money on this.
[1 (0.5405)]
Enter another sentence to parse or exit: Will cherish forever.
[2 (0.5639)]
```
# And thats it!!!
To view the source code and demo the project locally, please visit my [GitHub](https://github.com/uazhlt-ms-program/technical-tutorial-jailsalazar)!
# Related Links
To learn more about Flair, please visit the [Flair GitHub](https://github.com/flairNLP/flair)!

# Sources
- [Flair GitHub](https://github.com/flairNLP/flair)
- Akbik, A., Blythe, D., & Vollgraf, R. (2018). Contextual String Embeddings for Sequence Labeling. In COLING   2018, 27th International Conference on Computational Linguistics (pp. 1638–1649).
- Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., & Vollgraf, R. (2019). FLAIR: An easy-to-use framework for state-of-the-art NLP. In NAACL 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 54–59).
- [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/bittlingmayer/amazonreviews/activity)

