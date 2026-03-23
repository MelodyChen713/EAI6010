Text Classification Using ULMFiT and Traditional NLP Methods
Shaohua Chen
NUID:002300119

1. Dataset Selection and Business Problem
For this assignment, I selected the AG News dataset, which is a commonly used dataset for text classification tasks. The dataset contains news articles categorized into four classes: World, Sports, Business, and Sci/Tech. Each record contains a short text description of a news article and a corresponding label.
I chose this dataset because it is well-structured and widely used in natural language processing research. It is also suitable for comparing traditional NLP models with deep learning models. In addition, the dataset represents a realistic business scenario where organizations need to automatically classify large volumes of text data.
In real-world applications, automatic text classification can help businesses organize and manage information more efficiently. For example, news platforms and media companies need to categorize articles into different sections so that readers can easily find relevant content. Text classification is also useful for content recommendation systems, where articles are recommended based on user interests. Financial companies may also analyze news articles to track industry trends and economic developments.
Therefore, the AG News dataset provides a practical example of how machine learning can be used to automate text classification tasks in business environments.
2. Dataset Constraints
The original AG News dataset contains 120,000 training samples, which can be computationally expensive when training deep learning language models such as AWD_LSTM.
Since the assignment was implemented in Google Colab, computational resources were limited. To make the training process feasible, I randomly sampled 20,000 training examples from the original dataset. This reduced dataset still provided enough data to train both models while reducing the training burden.
The sampled dataset was divided into:
Training set (80%)
Validation set (20%)
This resulted in approximately 16,000 training samples and 4,000 validation samples. The original AG News test set was used for final evaluation.
These constraints allowed the models to be trained within the available computational resources while still maintaining meaningful experimental results.
3. Accuracy Comparison Between Models
Two different approaches were implemented for the text classification task.
The first approach was a traditional NLP model, which used TF-IDF vectorization combined with Logistic Regression. TF-IDF converts text into numerical features based on word importance, and Logistic Regression is used as the classifier.
The second approach was a deep learning model using the ULMFiT method. In this method, a pre-trained language model (AWD_LSTM) is first fine-tuned using the dataset. Then the learned encoder is transferred to a text classifier.
The results on the test dataset are shown below.
Model
Test Accuracy
Training Time
TF-IDF + Logistic Regression
0.9016
13.68 seconds
ULMFiT (AWD_LSTM)
0.9153
323.67 seconds


The results show that the ULMFiT deep learning model achieved higher accuracy than the traditional NLP model. The improvement is relatively small (about 1.4%), but it indicates that deep learning models can capture deeper semantic information in text.
4. Development Effort Comparison
Although the deep learning model achieved better accuracy, the development effort and computational cost were significantly higher.

Figure 1
The traditional NLP model was relatively simple to implement. The model only required two main steps: converting text into TF-IDF features and training a Logistic Regression classifier. The training process was very fast and completed in 13.68 seconds.
In comparison, the ULMFiT model required multiple stages and significantly more computational resources. The training process included:
Fine-tuning a pre-trained language model
Saving and loading the language model encoder
Training a text classifier
The language model fine-tuning took 146.27 seconds, and the classifier training took 177.39 seconds, resulting in a total training time of 323.67 seconds.
In addition to longer training time, the deep learning approach required more complex implementation. For example, the classifier needed to share the same vocabulary as the language model, and the encoder had to be transferred correctly between models.
Overall, the traditional NLP model required much less development effort and computational cost, while the deep learning model required more resources and implementation steps.
5. Recommended Model for Production
If this solution were deployed in a real production environment, the choice of model would depend on the system requirements.
If the goal is fast deployment and low computational cost, the traditional NLP model would be a good choice. It is simple, efficient, and already achieves high accuracy (over 90% in this experiment).
However, if the priority is maximum predictive performance, the ULMFiT model may be preferable. The deep learning model achieved the highest accuracy (91.53%) and can better capture contextual meaning in text.
In this experiment, the accuracy improvement from ULMFiT is relatively small compared to the large increase in training time and development complexity. Therefore, in many practical applications with limited resources, the traditional NLP model may be the more practical solution.
However, for large-scale systems where accuracy is critical and computational resources are available, deploying the ULMFiT model would be recommended.


References
Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL).
Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems.

