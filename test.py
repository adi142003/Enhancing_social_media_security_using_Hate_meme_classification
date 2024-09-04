import ollama

import glob
import pandas as pd
from PIL import Image

import os
from io import BytesIO

# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(filename):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['image_file', 'description'])
    return df

df = load_or_create_dataframe('image_descriptions.csv')

def get_png_files(folder_path):
    return glob.glob(f"{folder_path}/*.png")

# get the list of image files in the folder yopu want to process
image_files = get_png_files("./images") 
image_files.sort()

print(image_files[:3])
print(df.head())


# processing the images 
def process_image(image_file):
    print(f"\nProcessing {image_file}\n")
    with Image.open(image_file) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    full_response = ''
    # Generate a description of the image
    for response in ollama.generate(model='llava:34b', 
                             prompt="""Given the text embedded in the image [{text_in_image}], and the associated label for the image [{image_label}], provide a structured rationale for the image classification without explicitly mentioning the label. Your rationale should include the following categories, each within a 25-word limit:

Target Group or Person: Identify if the meme targets a specific individual, group, or demographic. Describe how the text or imagery in the meme addresses or represents them.

Content Evaluation: Assess if the content of the meme is potentially offensive or inappropriate. Explain any elements of the text or imagery that contribute to this assessment.

Context and Implications: Provide the context in which the meme is presented and how this context influences its perception. Discuss any implications that arise from the combination of text and image.

Overall Assessment: Summarize your reasoning by integrating the above categories to present a comprehensive understanding of the meme's impact.""",

                            #  prompt=f"can you create an AI model",
                             images=[image_bytes], 
                             stream=True):
        # Print the response to the console and add it to the full response
        # print(response['response'], end='', flush=True)
        full_response += response['response']

    print(full_response)
    # Add a new row to the DataFrame
    # df.loc[len(df)] = [image_file, full_response]


# for image_file in image_files:
#     if image_file not in df['image_file'].values:
#         process_image(image_file)

# # Save the DataFrame to a CSV file
# df.to_csv('image_descriptions.csv', index=False)
# ollama.pull('llava')
process_image("./DATA/trainImages/7.jpg")

