# Chapter 05 - Insights and Interpretability

In this chapter we investigate some ideas for understanding how the language models represent
semantics inside their embedding vectors.

We explore techniques for both superivised (labelled) data problems and unsupervised (unlabelled) data problems. 


## Principal Components

In the first section we use the principal components technique to project the semantic vector into a lower number of dimensions so we can see where our individual data points are distributed.
 
* [CaseStudy_5.1_01_Principal_Components.ipynb](CaseStudy_5.1_01_Principal_Components.ipynb)

## KeyBERT

We then use the KeyBERT package to extract semantically informed keywords and phrases from the data.

* [CaseStudy_5.1_02_KeyBERT.ipynb](CaseStudy_5.1_02_KeyBERT.ipynb) 
* [CaseStudy_5.1_03_KeyBERT_Topic.ipynb](CaseStudy_5.1_03_KeyBERT_Topic.ipynb)

## Feature Importance

Finally we look at a variety of feature importance methods applied to language models. The goal of which is to understand which parts of the semantic vector are driving performance, and which aspects of the text are influencing model predictions.

* [CaseStudy_5.2_01_Feature_Importance.ipynb](CaseStudy_5.2_01_Feature_Importance.ipynb)
* [CaseStudy_5.2_02_SHAP.ipynb](CaseStudy_5.2_02_SHAP.ipynb)
* [CaseStudy_5.2_03_Permutation.ipynb](CaseStudy_5.2_03_Permutation.ipynb)
 
