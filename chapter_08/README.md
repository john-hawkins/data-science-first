# Chapter 08 - Synthetic Data Generation

In this chapter we explore the idea of using language models to generate data for niche machine learning
projects. The core idea is to understand how we can ensure that we get sufficient variation that is 
representative of the problem we want to solve.

We used `uv` to configure the project so the examples could be easily executed.


## Case Study 8.1 - Customer Complaint Routing

In the first case study we want to build a system that can recognise different types of customer complaints.
The case study focuses on a start-up that wants to build a general product for the telecommunications industry.
We use our generative models to emulate different categories of customer complaint that are common for telcos.

Use the [Jupyter Notebook](CaseStudy_8.1_01_Complaints.ipynb) to step through the fundamental ideas in this case study.
That notebook makes uses of the content of the indivual code listings shown in the book:

* [CaseStudy_8.1_01.py](CaseStudy_8.1_01.py)
* [CaseStudy_8.1_02.py](CaseStudy_8.1_02.py)
* [CaseStudy_8.1_03.py](CaseStudy_8.1_03.py)
* [CaseStudy_8.1_04.py](CaseStudy_8.1_04.py)

The final script for generating the complaint emails can be found here: [generate_complaint_emails.py](generate_complaint_emails.py)


For the visualisation and analysis of the generated data we used a script to create Instruct embeddings,
this script can be found here: [generate_embeddings.py](generate_embeddings.py)

We then visualised and audited the generated data in the final listsings of the case study:

* [CaseStudy_8.1_05.py](CaseStudy_8.1_05.py)
* [CaseStudy_8.1_06.py](CaseStudy_8.1_06.py)

## Case Study 8.2 - Idiom Translation

In the second case study we want to build a system for a publishing company that can recognise common problems
in the translaton of books for foreign markets. We use the generative capacities of multi-language models to
build up a dataset that exemplifies these specific idiom translation problems.
 
You can step through the code and a small amount of documentation using the Jupyter Notebook file:
[CaseStudy_8.2_01_Idioms.ipynb](CaseStudy_8.2_01_Idioms.ipynb). Alternatively, you can go through raw
python scripts:

* [CaseStudy_8.2_01.py](CaseStudy_8.2_01.py)
* [CaseStudy_8.2_02.py](CaseStudy_8.2_02.py)
* [CaseStudy_8.2_03.py](CaseStudy_8.2_03.py)
* [CaseStudy_8.2_04.py](CaseStudy_8.2_04.py)
* [CaseStudy_8.2_05.py](CaseStudy_8.2_05.py)
   
After the dataset has been created we then introduced the use of BERT models for sentence pair classification
as shown in the second notebook: [CaseStudy_8.2_02_BERT.ipynb](CaseStudy_8.2_02_BERT.ipynb)

* [CaseStudy_8.2_06.py](CaseStudy_8.2_06.py)
* [CaseStudy_8.2_07.py](CaseStudy_8.2_07.py)
* [CaseStudy_8.2_08.py](CaseStudy_8.2_08.py)

### Errata: Missing Figure in Chapter 8
 
There is a missing figure in this chapter. The image displayed as Figure 8.3 should actually be the 
confusion matrix shown below. 

[!Few-Shot Classifier Confusion Matrix](results/few_shot_cm.png)

The figure labelled as 8.3 in the book should be labelled Figure 8.4, the later reference to Figure 8.3 
is for the precision recall curve for the fine-tuned BERT model, shown below.

[!Idiom Classifier Precision-Recall Curve](results/idiom_transalation_PRC.png)


