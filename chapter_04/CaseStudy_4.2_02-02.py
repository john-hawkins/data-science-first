from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Use Pydantic to define the data structure
class Topic(BaseModel):
    name: str = Field(description="The name of the core topic being discussed in all provided essays.")

structured_model = model.with_structured_output(Topic)


DELIMITER = '####'

def generate_prompt_text(essays: list[str]) -> str:
    TEXT_DELIMITER = "\n" + DELIMITER + "\n"
    essays_comb = TEXT_DELIMITER.join(essays)
    user_message = f'''
    Below is a set of student essays delimited with {DELIMITER}.

    Please identify the single main topic discussed in these essays. 
    Return a just a topic name for the complete set.
    The topic name should be short, between one and three words long. 

    Student Essays
    {DELIMITER}
    {essays_comb}
    {DELIMITER}
    '''
    return user_message


def get_topic_label(essays):
    prompt = generate_prompt_text(essays)
    result = structured_model.invoke(prompt)
    return result.name

import numpy as np
import pandas as pd

df = pd.read_csv("data/clustered_data.csv")

clusters = df['cluster'].unique().tolist()
clusters.sort()

results = pd.DataFrame(columns=('ID', 'Records', 'Topic Name', 'Mean Dist'))

topics = {}

for c in clusters:
    temp = df[df['cluster']==c].copy()
    temp = temp.sort_values("dist_to_centroid").reset_index()
    examples = temp.loc[0:4,'text'].tolist()
    topic = get_topic_label(examples)
    topics[c] = topic
    record = {'ID':c, 'Records':len(temp), 'Topic Name':topic, 'Mean Dist':temp["dist_to_centroid"].mean()}
    results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)


# Save the results and Display as a Markdown Table
results.to_csv("topic_labels", index=False)
markdown_table = results.to_markdown(index=False)
print(markdown_table)

