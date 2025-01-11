# Paranormal-Activity-Classification-with-Machine-Learning-and-NLP
This repository contains the code, data, and resources for the project Classifying Paranormal Activities with Machine Learning and Text-Based Deep Learning Models. The goal is to classify textual descriptions of paranormal activities into specific categories using various machine learning and deep learning techniques.

# Summary
Leveraged machine learning and transformer-based deep learning models to analyze over 14,000 textual descriptions of paranormal events. Starting with traditional machine learning models, we progressively adopted more complex models like BERT and XLNet, achieving state-of-the-art results.

Key features of the dataset include:
Descriptions of events across folklore, paranormal activity, and cryptozoology.
Rich narrative data scraped from The Paranormal Database.

# Key Components
1. Dataset: Textual data scraped using BeautifulSoup from The Paranormal Database.
2. Preprocessing: Duplicate removal, text standardization, punctuation, and stop-word removal. Feature extraction using TF-IDF and Bag of Words.
3. Models:
   Traditional ML: Random Forest, LightGBM, SVC.
   Deep Learning: BERT, XLNet, Neural Attention Forest.
4. Results:
BERT achieved the highest accuracy of 91.54%.
Random Forest and LightGBM followed closely with ~90% accuracy.

Results

Model	            Accuracy	Precision	Recall	F1-Score
BERT	            91.54%	   92%	    91%	     91.5%
Random Forest	    90.15%	   90%	    89.5%	   89.8%
LightGBM	        90.26%	   90.3%	  90%	     90.1%

5. Future Work
   A. Integrating Bag-of-Words with Transformers:Explore combining traditional bag-of-words features with transformer embeddings to enhance contextual representation and improve model interpretability. This hybrid approach aims to leverage both the simplicity of bag-of-words and the nuanced contextual understanding of transformers.

   B. Combining Bag-of-Words and TF-IDF for Deep Learning and ML Models: Investigate the synergistic use of bag-of-words and TF-IDF features as input for both deep learning and machine learning models. This integration seeks to capture complementary information from both global term frequencies and local text structure, improving classification performance across diverse models.
