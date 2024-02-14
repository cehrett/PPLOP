from utils.load_llm_model import prepare_to_load_model
prepare_to_load_model('cehrett')
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import argparse
import glob
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up argument parser
parser = argparse.ArgumentParser(description="Script to generate and evaluate system prompts.")

# Adding argument definitions
parser.add_argument("--labeled_data_loc", type=str, required=True, help="Location of the labeled data CSV file.")
parser.add_argument("--base_system_prompt_file", type=str, default=os.path.join('.', 'base_system_prompt.txt'), help="File containing the base system prompt text.")
parser.add_argument("--meta_system_prompt_file", type=str, default=os.path.join('.', 'meta_system_prompt.txt'), help="File containing the meta system prompt text.")
parser.add_argument("--already_begun", type=bool, default=False, help="Flag indicating whether the process has already been started.")
parser.add_argument("--generated_system_prompts_loc", type=str, default=os.path.join('.', 'outputs', 'system_prompt_scores_simple_ppl.csv'), help="Location to save or load generated system prompts and their scores.")
parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Hugging face model id for the LLM to be used as both scorer and prompt generator.")

# Parse arguments
args = parser.parse_args()

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype=torch.float16)

# User inputs:
generated_system_prompts_loc = os.path.join('.', 'outputs', 'system_prompt_scores_simple_ppl.csv')


# Get base and meta system prompts
# Initialize an empty list to store the contents of each txt file
base_system_prompts = []

# Use glob to find all txt files in the base_system_prompts folder
for filename in glob.glob('base_system_prompts/*.txt'):
    with open(filename, 'r', encoding='utf-8') as file:
        base_system_prompt = file.read()
        # Add the contents of each file to the list
        base_system_prompts.append(base_system_prompt)


# Read the meta system prompt from a text file
with open(args.meta_system_prompt_file, 'r', encoding='utf-8') as file:
    meta_system_prompt = file.read()


# Load data
df_med_text = pd.read_csv(args.labeled_data_loc)
labels = df_med_text.label.unique().tolist()
   

# Helper functions
def make_fsl_prompt(system_prompt, text, df=df_med_text, n=5):
    """
    Generate a Few-Shot Learning (FSL) prompt by sampling examples from a DataFrame and appending a new text to be labeled.

    This function filters out rows from the provided DataFrame where the 'text' column matches the specified 'text' parameter.
    It then randomly samples 'n' rows from the filtered DataFrame to create FSL examples. These examples, along with the new 
    text to be labeled, are formatted and appended to the given 'system_prompt'.

    Parameters:
    - system_prompt (str): The initial system prompt to which the FSL examples and new text will be appended.
    - text (str): The text to be labeled, which is also used to filter out matching rows from the DataFrame (so the FSL examples don't include the text to be labeled)
    - df (pd.DataFrame): The DataFrame from which FSL examples are sampled. Defaults to 'df_med_text'.
    - n (int): The number of FSL examples to sample from the DataFrame. Defaults to 5.

    Returns:
    - str: The complete FSL prompt consisting of the initial system prompt, FSL examples, and the new text to be labeled.

    Example:
    - Given a system prompt "Please classify the following texts:", a text "New medical study results", and a DataFrame of 
      medical texts with labels, the function will return a prompt with the system prompt, five randomly chosen medical texts 
      with labels, and the new text to be classified.
    """
    # Randomly sample n samples with labels, for FSL. Exclude rows matching text.
    filtered_df = df[df['text'] != text]  # Filter to get rows where df.text does not match the string 'text'
    sampled_df = filtered_df.sample(n=n)
    
    # Make FSL section of prompt
    fsl_examples = [
        item for i in range(n)
        for item in [
            {"role": "user", "content": sampled_df.text.iloc[i]},
            {"role": "assistant", "content": sampled_df.label.iloc[i]}
        ]
    ]
    
    # Since Mistral doesn't have a dedicated system prompt, add ours to the first user message
    intermediary_text = "\n\n# Begin text:\n"
    fsl_examples[0]["content"] = system_prompt + intermediary_text + fsl_examples[0]["content"]
        
    # Make final section of prompt with text to be labeled
    to_be_labeled_prompt = [{"role": "user", "content": text}]
    
    return fsl_examples + to_be_labeled_prompt


def get_ppl_of_completion(prompt, labels=labels, model=model, tokenizer=tokenizer, device=device):
    """
    Calculates the perplexity for each label appended to the given prompt using a specified language model.

    This function takes a prompt and a list of labels, then for each label, it calculates the perplexity of the 
    prompt when that label is appended. Perplexity is computed by encoding the prompt with each label, generating 
    predictions using a language model, and then calculating the mean of the negative log-likelihood (NLL) loss 
    to get the perplexity.

    Parameters:
    - prompt (str): The initial prompt to which labels will be appended.
    - labels (list of str): A list of labels to be appended to the prompt. Defaults to the unique values of the labels in the provided data.
    - model (transformers.PreTrainedModel): The pre-trained language model used for generating predictions.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model, used for encoding text.
    - device (torch.device): The device (e.g., CPU, GPU) on which the model computations will be performed.

    Returns:
    - dict: A dictionary with labels as keys and their corresponding perplexity values as values.

    Example:
    - Given a prompt "The climate change impact is", a list of labels ["positive", "negative", "neutral"], and a 
      specified language model, the function will return a dictionary with the perplexity of each label when appended 
      to the prompt, like {"positive": 23.4, "negative": 30.1, "neutral": 27.6}.
    """
    output_ppls = {}
    
    for label in labels:
        # Get the message corresponding to the assistant returning this label
        label_message = [{"role": "assistant", "content": label}]
        
        # Encode the initial prompt
        desired_encodings = tokenizer.apply_chat_template(prompt + label_message, return_tensors='pt')
        desired_input_ids = desired_encodings.to(device)
        desired_target_ids = desired_input_ids.clone()
        desired_target_ids[:, :len(tokenizer.apply_chat_template(prompt))] = -100

        # Generate predictions from the model
        with torch.no_grad():
            # Here we are using the model to generate outputs without providing labels
            desired_outputs = model(desired_input_ids, labels=desired_target_ids)

            # Get the predictions from the last layer
            desired_predictions = desired_outputs.logits
            desired_nll = desired_outputs.loss

        # Get the perplexity of the final tokens (corresponding to the label)
        ppl = torch.exp(desired_nll.mean())
        output_ppls[label] = round(ppl.cpu().item(),2)
    
    return output_ppls


def get_meta_prompt(df_system_prompt_scores, meta_system_prompt, n=8):
    """
    Generates a meta prompt that includes a table of previous system prompts and their scores, formatted in markdown.

    This function takes a DataFrame containing system prompts and their associated scores, and formats it into a markdown 
    table. This table is then appended to a provided meta system prompt. The purpose is to create a comprehensive meta 
    prompt that includes historical data about previous prompts and their performance.

    Parameters:
    - df_system_prompt_scores (pd.DataFrame): A DataFrame containing system prompts and their associated scores. 
      It should have columns that represent different aspects of the prompts and scores.
    - meta_system_prompt (str): The initial meta system prompt to which the markdown table will be appended.

    Returns:
    - str: The complete meta system prompt, which includes the initial meta system prompt followed by a markdown table 
      of previous system prompts and their scores.

    Example:
    - Given a DataFrame with columns ["Prompt", "Score"] and a meta system prompt "Overview of System Performance:", 
      the function will return a string that starts with "Overview of System Performance:", followed by a markdown 
      table of the prompts and scores from the DataFrame.
    """
    # Format previous system prompts and scores into the meta prompt
    ## Group by 'system_prompt' and calculate the mean of 'score' for each group
    deduplicated_df = df_system_prompt_scores.groupby('system_prompt', as_index=False)['score'].mean()
    # Sort by 'score' in ascending order
    sorted_df = deduplicated_df.sort_values(by='score', ascending=True)

    # Get n//2 rows with the lowest scores
    lowest_scores_df = sorted_df.head(n//2)

    # If there are enough rows, randomly sample the rest; otherwise, sample from what's available
    if len(sorted_df) > n//2:
        random_sample_df = sorted_df.iloc[n//2:].sample(frac=1).head(n-n//2)
    else:
        random_sample_df = pd.DataFrame()  # If not enough data, use an empty DataFrame

    # Combine the two sets of data
    final_df = pd.concat([lowest_scores_df, random_sample_df])
    
    # Sort the final version
    final_df = final_df.sort_values(by='score', ascending=False)

    # Create the list of prompts
    system_prompts_list = [
        item for _, row in final_df.iterrows()
        for item in [
            {"role": "assistant", "content": row['system_prompt']},
            {"role": "user", "content": f"Score: {row['score']:.2f}. Try again with a new system prompt."}
        ]
    ]
        
    # Make the prompt
    meta_prompt = [{"role": "user", "content": meta_system_prompt}] + system_prompts_list
    
    return meta_prompt


def get_new_system_prompt(meta_prompt, model=model, tokenizer=tokenizer, device=device):
    prompt_input_ids = tokenizer.apply_chat_template(meta_prompt, return_tensors='pt')
    output = model.generate(prompt_input_ids.to(device), max_new_tokens=400, do_sample=True)
    
    system_prompt = tokenizer.decode(output[0][len(prompt_input_ids[0]):-1])
    
    return system_prompt


# This function is intended to be called in a .apply statement on the df.
def get_ppl_for_row(row, system_prompt):
    """
    Calculates the average perplexity for a given text row using a system prompt.

    This function takes a row from a DataFrame and a system prompt, generates a complete prompt using the text from the row, 
    and then calculates the perplexities for this prompt. The perplexities are calculated for predefined labels (as defined 
    in `get_ppl_of_completion`). The function returns a Pandas Series containing these perplexities, which can be used to 
    add new columns to the original DataFrame.

    Parameters:
    - row (pd.Series): A row from a DataFrame. The row should contain a 'text' field with the text to be analyzed.
    - system_prompt (str): The initial system prompt to which the text will be appended to form a complete prompt.

    Returns:
    - pd.Series: A series where each index is a label and the corresponding value is the perplexity of the prompt with that label.

    Note:
    - This function is designed to be used with the DataFrame.apply() method, passing axis=1.
    - The 'make_fsl_prompt' and 'get_ppl_of_completion' functions need to be defined in the scope where this function is used.
    """
    # Get the text to be labeled
    text = row.text
    
    # Make prompt
    prompt = make_fsl_prompt(system_prompt, text)
    
    # Get perplexities
    ppls = get_ppl_of_completion(prompt)
    
    return pd.Series(ppls)
        
    
def get_avg_ppl_of_correct_label(system_prompt, df=df_med_text):
    # Update user
    print('Running get_avg_ppl_of_correct_labels. About to get ppl for each row for each label.')
    
    # Get the ppl for each row for each label
    df_ppls_for_each_label = df.join(df.apply(get_ppl_for_row, axis=1, system_prompt=system_prompt))
    
    # Get the ppl of the correct label for each row, minus the ppls of the other labels
    label_possibilities = df['label'].unique()  
    df_ppls_for_each_label['correct_label_ppl'] = \
        df_ppls_for_each_label.apply(lambda row: row[row['label']], axis=1) #  -np.mean([row[label] for label in label_possibilities if label != row['label']])

    # Get the average ppl of the correct label
    avg_ppl = round(df_ppls_for_each_label['correct_label_ppl'].mean(),2)
    
    return avg_ppl


# Assumes df_system_prompt_scores at least has one row for the base system prompt.
def search_system_prompt_space(meta_system_prompt, df_system_prompt_scores=None, system_prompt=None, model=model):
    if df_system_prompt_scores is None: 
        df_system_prompt_scores = pd.DataFrame(columns=['system_prompt','score'])
    else:
        # Make a new meta_prompt using the most recent version of df_system_prompt_scores
        meta_prompt = get_meta_prompt(df_system_prompt_scores, meta_system_prompt)

        # Get new system_prompt by showing meta_prompt to model
        system_prompt = get_new_system_prompt(meta_prompt)
    
    # Check to make sure we have a system_prompt
    assert system_prompt is not None, "If no df_system_prompt_scores is provided, a system_prompt must be"
    
    # Use get avg ppl using new system_prompt
    avg_ppl = get_avg_ppl_of_correct_label(system_prompt)
    
    # Create a new DataFrame with the new row
    new_row = pd.DataFrame({'system_prompt': [system_prompt], 'score': [avg_ppl]})

    # Append the new DataFrame to the existing one
    df_system_prompt_scores = pd.concat([df_system_prompt_scores, new_row], ignore_index=True)
    
    return df_system_prompt_scores


if args.already_begun:
    # Load the previously generated prompts and scores
    df_system_prompt_scores = pd.read_csv(args.generated_system_prompts_loc)
else:
    # Create initial prompts and scores, using the base_system_prompts
    df_system_prompt_scores_list = []
    for base_system_prompt in base_system_prompts:
        new_df_system_prompt_scores_row = search_system_prompt_space(meta_system_prompt, system_prompt=base_system_prompt, df_system_prompt_scores=None)
        df_system_prompt_scores_list.append(new_df_system_prompt_scores_row)
    df_system_prompt_scores = pd.concat(df_system_prompt_scores_list, ignore_index=True)
    # Save results so far
    df_system_prompt_scores.to_csv(args.generated_system_prompts_loc, index=False)

# Loop continuously
while True:
    # If user has screated this file, then stop looping
    if os.path.exists('stop_script.flag'):
        print("Stop flag detected. Stopping the script.")
        os.remove('stop_script.flag')  # Automatically delete the flag file
        break
        
    df_system_prompt_scores = search_system_prompt_space(meta_system_prompt, df_system_prompt_scores=df_system_prompt_scores, system_prompt=None)
    
    # Save results so far
    df_system_prompt_scores.to_csv(args.generated_system_prompts_loc, index=False)
    
    # Output most recent prompt and score
    print(f'NEW PROMPT:\n{df_system_prompt_scores.iloc[-1]["system_prompt"]}\nSCORE:\n{df_system_prompt_scores.iloc[-1]["score"]}\n\n')
    
    
