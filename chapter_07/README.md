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

Processing all of the records is doe with the CLI script:

```
uv run python process_json_files.py free-news-datasets/law ./law_test.csv 
uv run python process_json_files.py free-news-datasets/biz ./biz_test.csv
``` 

Analysis of those results is done as follows:

```
uv run python analyse_results.py ./law_test.csv ./biz_test.csv
```


## Case Study 7.2 - Feature Engineering

In the second case study we explore the idea of using a generative language model to create features for a dataset.
The case study focuses on a education technology application in which we look at creating features that will be used across multiple models that score
student essays for various aspects of writing quality.

Use the [Jupyter Notebook](CaseStudy_7.2_01_Features.ipynb) to step through the fundamental ideas in this case study.

The complete set of code is compiled into a [single command line application](essay_features.py) to process a dataset of essays and produce all features for each record in the data. That script can be invoked as follows:

```
uv run python essay_features.py data/Essays.csv data/Essays_scored.csv
```

You can then evaluate the features using a script that looks at correlation between our generated features and the established quantitative methos of evaluating writing. Ideally you want to see some correlation (indicting coherenece in what is being measured), but also sufficient independence to indicate that you features could have novel information value.

```
uv run python analyse_essays.py data/Essays_scored.csv
```

