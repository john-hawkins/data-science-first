# Chapter 07 - Labelling and Feature Engineering

In this chapter we explore the idea of using language models with tasks preparing to build a bespoke
machine learning solution. We start by using a model to help label a dataset for us, and then use the
model to generate features we can test in the model development process.

We used `uv` to configure the project so the examples could be easily executed.


## Case Study 7.1 - Labelling Industry Verticals

In this case study we take a dataset of news reports and look to label them with the most appropriate category
from the NAICS industry categroisation schema. We apply few shot prompting to the task of labelling the data
and then auditing it to refine the labels.

You can simply start the notebook with the command:

```
uv run jupyter notebook CaseStudy_7.1_01_Labelling.ipynb  
```

## Case Study 7.2 - Feature Engineering


