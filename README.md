# Tweet Text Sentiment Analysis Project

## Project Overview

This project aims to develop an advanced Natural Language Processing (NLP) model for analyzing sentiment in Tweets related to Apple and Google products. The dataset comprises over 9,000 Tweets labeled as positive, negative, or neutral by human raters. The objective is to create a robust generalized sentiment classification model that can accurately predict sentiment based on the content of the Tweets, providing Apple and Google with actionable insights for their marketing and product development teams. The goal is to help these stakeholders understand consumer perception, identify trends, and enhance customer engagement.

### Objectives:
1. **Develop a Sentiment Classification Model**:
   - Build an NLP model capable of accurately classifying Tweets as positive, negative, or neutral to help Apple and Google assess public opinion about their products.
2. **Evaluate Model Performance for Binary and Multiclass Classification**:
   - Optimize binary classification (positive vs. negative) and multiclass classification (positive, negative, neutral) to ensure high accuracy and balance across all classes, with a focus on improving the detection of negative sentiment.
3. **Provide Business Insights to Stakeholders**:
   - Use model results to provide insights on how Apple and Google can respond to public opinion, identify marketing opportunities, and enhance product development** strategies.

The models developed can be deployed to provide actionable insights for stakeholders at Apple and Google, ensuring they can monitor public sentiment effectively and make informed decisions for marketing and product development.

## Business Understanding

* **Stakeholder:** The primary stakeholders are the marketing and product development teams at Apple and Google. These teams seek to understand public sentiment towards their products to guide marketing strategies, product enhancements, and customer engagement initiatives.

* **Business Problem:** Apple and Google aim to monitor and analyze general consumer sentiment on social media platforms to gauge public perception and identify trends in how their products are received. A generalized sentiment analysis model based on Tweet content will help them quickly assess whether social media conversations about their products are positive, negative, or neutral. This will allow them to respond to broad public opinion more effectively, identify potential marketing opportunities, and enhance customer engagement.

## Data Understanding

* **Dataset:** The project utilizes a dataset from CrowdFlower via data.world, containing over 9,000 Tweets about Apple and Google products. Each Tweet annotated by human raters with a sentiment classification: positive, negative, or neutral.

* **Suitability:** This dataset is well-suited for the business problem as it provides a sizable and diverse collection of real-time consumer opinions. The multi-class sentiment labels facilitate both binary (positive/negative) and multi-class classification tasks, allowing for scalable model development.

## Data Exploration & Preparation

The dataset initially contained 9,093 entries and three key columns: tweet_text, which represents the actual Tweet content, emotion_in_tweet_is_directed_at, and is_there_an_emotion_directed_at_a_brand_or_product, which holds the sentiment labels. Upon inspection, it was determined that the emotion_in_tweet_is_directed_at column had a large number of missing values (5,802), making it irrelevant for our primary task of sentiment classification, so it was dropped. Additionally, one row had a missing tweet_text value and was removed. The sentiment distribution showed a significant imbalance, with the majority of the tweets classified as "No emotion toward brand or product," followed by positive and negative emotions, highlighting the need for class-balancing techniques later in the pipeline.

During exploration, tweet length was analyzed, revealing most tweets to be of moderate length, which aligns with the character constraints of Twitter. A word cloud was generated to visualize the most frequent terms, showing that many tweets referenced event-specific terms like "SXSW" and included non-informative elements such as mentions and links. This prompted the decision to refine the text through pre-processing by converting text to lowercase, removing punctuation, tokenizing, removing stopwords, and lemmatizing to reduce words to their base form.

After cleaning and pre-processing, the data was prepared for sentiment classification. The sentiment labels were encoded appropriately for the binary classification and later for the multiclass classification. The data was split into an 80/20 train-test set, and a TF-IDF vectorizer was applied to transform the textual data into numerical features. Given the class imbalance in the dataset, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to balance the classes and ensure that sentiments were adequately represented for the machine learning models. This prepared the data for building robust sentiment classification models.

## Modeling

### Modeling Summary

In this project, multiple binary classification models were trained to predict sentiment in Tweets about Apple and Google products. The models included **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)**. After tuning, **Logistic Regression** emerged as the best overall performer, achieving an accuracy of **88%**. Its strength lay in its ability to detect negative sentiments, with a recall of **0.61** and an F1-score of **0.62**—both outperforming SVM and Random Forest. While **SVM** had a slightly higher accuracy of **89%**, its lower recall and F1-score for negative sentiments made it less reliable for capturing this challenging class. Random Forest, with its recall of **0.36**, struggled similarly. Therefore, **Logistic Regression** was chosen as the most balanced model for binary sentiment analysis.

For the multiclass sentiment classification task, two models were trained: **Multinomial Naive Bayes** and an **ensemble model** combining Multinomial Naive Bayes and Random Forest. The ensemble model achieved superior results, with an accuracy of **68%**, compared to Naive Bayes' **61%**. It performed better across all sentiment classes, particularly in improving the detection of negative sentiments and capturing neutral emotions. With a more balanced performance across positive, negative, and neutral classes, the ensemble model proved to be the preferred choice for multiclass sentiment classification, offering more reliable results for real-world applications.

## Evaluation Metrics

The evaluation metrics used so far include Accuracy (proportion of correctly classified instances (both positive, negative, and neutral sentiments) out of the total instances), Precision (ratio of correctly predicted positive instances to the total instances predicted as positive), Recall (model's ability to identify all relevant instances of a class) and F1-Score (harmonic mean of precision and recall, providing a balanced metric when both precision and recall are important). The higher the metric, the better.

In the context of generalized sentiment analysis, **F1-score** would be the most important metric to consider, as it balances both **precision** and **recall**. Given that the dataset includes several classes, it is crucial not only to correctly classify the sentiments (precision) but also to capture as many relevant examples of each class as possible (recall). F1-score helps ensure that the model performs well across different classes without favoring one over the others, which is particularly important in cases of class imbalance like this one. Thus, while accuracy provides a general overview, the F1-score offers a more nuanced view of model performance across all sentiment categories.

We also included a confusion matrix (to show the true and predicted classes for each category) and AUC-ROC (to plot the true positive rate (recall) against the false positive rate at various threshold levels) charts for best performing models in each classification category.

## Conclusion

The results of this project demonstrate the successful development of an NLP sentiment classification model aimed at analyzing consumer sentiment toward Apple and Google products on Twitter. For binary classification, Logistic Regression proved to be the most balanced model, achieving an accuracy of 88%, with a particularly strong performance in detecting negative sentiment. This is critical for stakeholders since negative sentiment often highlights areas for improvement in products and services. In the multiclass classification task, the ensemble model (Multinomial Naive Bayes + Random Forest) outperformed the standalone Naive Bayes model, achieving 68% accuracy with a well-balanced performance across positive, negative, and neutral sentiment classes. This indicates that the model can reliably provide actionable insights across all sentiment categories, helping Apple and Google make informed decisions about their products and customer engagement.

The project successfully met its primary objectives by building robust general sentiment classification models that can be deployed to provide relevant insights to the marketing and product development teams at Apple and Google. The models were able to capture both positive and negative sentiments, with a special emphasis on improving the detection of negative sentiment, which is vital for addressing public concerns and enhancing product strategies. The ensemble model's superior performance in the multiclass task further ensures that neutral sentiments are well-handled, providing a comprehensive view of consumer opinions.

### Recommendations:

Based on the results and the nature of the data, the next steps should focus on:

- **Improve Detection of Negative Sentiment**: Focus on enhancing the model’s ability to capture negative sentiments by exploring advanced techniques like **phrase extraction** or more sophisticated models such as BERT to capture nuanced expressions of negativity.
  
- **Address Class Imbalance**: Consider collecting more diverse data to better balance the representation of sentiment classes, particularly negative sentiment, which is often underrepresented in the dataset.

- **Refine Pre-processing**: Implement more detailed pre-processing techniques (e.g., handling negations, domain-specific words) to improve model performance across all sentiment classes.

- **Ongoing Monitoring and Retraining**: Regularly monitor and retrain the model as the social media landscape and consumer sentiment evolve, ensuring that the model continues to provide accurate insights over time.