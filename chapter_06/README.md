# Chapter 06 - Zero to Few Shot Prompting

In this chapter we explore the idea of turning a language model into a classifier using the techniques
of zero and few shot prompting. These examples will require access to proprietary LLM vendors APIs.

You should be able to easily switch between vendors by changing the library used to import and initiate the model.

**NOTE:** All code examples assume you have your API set up in your environment variables.


## Case Study 6.1 - Extracting and Classifying

In the first case study we look at processing a dataset of company reports to extract 
information about the company and classify the reports into one of three tiers of sentiment.

The complete process is captured in the (Notebook CaseStudy_6.1_01_Classification.ipynb)[CaseStudy_6.1_01_Classification.ipynb]
 and you can see the raw code listings from the book in the python scripts:

* [CaseStudy_6.1_01-01.py](CaseStudy_6.1_01-01.py)
* [CaseStudy_6.1_01-02.py](CaseStudy_6.1_01-02.py)


## Case Study 6.2 - Evaluating Resumes According to Selection Criteria

In the second case study we expand on these ideas to extract selection criteria from resume documents
and then ask a model to evaluate how well each candidate matches the selection criteria.

The complete process is outlined in the (Notebook file CaseStudy_6.2_01_Resume_Sorting.ipynb)[CaseStudy_6.2_01_Resume_Sorting.ipynb], 
and the individual scripts contain the code listings from the published book:

* [CaseStudy_6.2_01-01_extract_text.py](CaseStudy_6.2_01-01_extract_text.py)
* [CaseStudy_6.2_01-02.py](CaseStudy_6.2_01-02.py)
* [CaseStudy_6.2_01-03.py](CaseStudy_6.2_01-03.py)

