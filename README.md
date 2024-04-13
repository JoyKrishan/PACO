# PaCo
==============================

Automated program repair promises to streamline the debugging process and reduce manual effort. However, a common problem known as overfitting hinders the success of current APR methods. To tackle this challenge, we introduce PaCo, a novel end-to-end framework for automated patch correctness assessment. PaCo combines a code representation model with a patch correctness classifier. Our code representation model uses contrastive learning to generate meaningful code embeddings, drawing correct patches closer to their buggy counterparts while distancing incorrect patches. The patch correctness classifier then analyzes these embeddings to determine if a patch represents a valid fix. Our approach enhances the accuracy of correct patch identification, potentially minimizing the need for time-consuming test-case execution or less reliable static analysis. On a small, balanced dataset, PaCo demonstrates promising results, achieving an **F1-score of 95.3** and an **AUC of 0.97**. However, its performance highlights the challenges posed by large, imbalanced datasets. Our study also underscores the importance of balanced datasets for effective patch correctness assessment, as they enable models to learn more discriminative features.

### Results
------------
| Classifiers | set  | Acc. | Prec. | Rec. | F1  | AUC |
|-------------|------|------|-------|------|-----|-----|
| LR          | small| 95.0 | 92.4  | 98.4 | 95.3| 0.97|
| DT          |      | 88.2 | 91.4  | 85.5 | 88.3| 0.88|
| RF          |      | 94.1 | 93.7  | 95.2 | 94.4| 0.96|
| LR          | large| 99.5 | 99.5  | 99.4 | 99.5| 0.99|
| DT          |      | 99.1 | 98.8  | 99.4 | 99.1| 0.99|
| RF          |      | 99.5 | 99.4  | 99.6 | 99.5| 0.99|



*Table 1: Performance Comparison of Patch Classifiers in the PaCo Framework Across Different Datasets*


| Code Encoders    | Learners | F1 | AUC   | 
|----------|----------|----------|------|
| BERT     | LR       | 72.0     | 0.81 |      
|          | DT       | 59.6     | 0.63 |      
|          | NB       | 64.5     | 0.64 |      
| CC2Vec   | LR       | 72.0     | 0.78 |      
|          | DT       | 67.2     | 0.69 |      
|          | NB       | 28.5     | 0.72 |      
| Doc2Vec  | LR       | 62.3     | 0.71 |      
|          | DT       | 57.7     | 0.60 |      
|          | NB       | 57.9     | 0.71 |      
| CACHE    |          | 78.0     | 78.0 | 
| PaCo     | LR       | 95.3     | 0.97 |      
|          | DT       | 88.3     | 0.88 |   


*Table 2: Comparison of PaCo with Other Techniques on Small-Scale Dataset*



