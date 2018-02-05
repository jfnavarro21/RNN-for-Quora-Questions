# RNN-for-Quora-Questions
Based on Kaggle's Quora question pairs   

This project works with data from the Kaggle Competition called Quora Question Pairs. 

## Project Description
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

## Solution Description
This project was tackled in two parts. The first part was to use natural language processing techniques to find numerical representations of questions. Various machine learning methods were applied on these questions to determine similarity of the paris of questions. The feature engineering and modeling was done on the University of Chicago's Spark cluster.

The second portion of this project was to build an RNN architecture to achieve a higher accuracy of predictions. This RNN was built using an LSTM layer in Keras. The data was analyzed on the University of Chicago's GPU cluster.
