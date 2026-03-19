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

