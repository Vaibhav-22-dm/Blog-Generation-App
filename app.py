import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

## Function to get response from LLAMA 2 model
def getLlamaResponse(input_text, no_words, blog_style):
    
    ### Llama2 model
    llm = CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
            }
        )

    ### Prompt Template
    template = """
        Write a blog for {blog_style} on the topic: {input_text} 
        within {no_words} words.
        """

    prompt = PromptTemplate(
       input_variables=["blog_style", "input_text", "no_words"],
       template=template
       )
    
    ### Generate the response from the Llama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)

    return response


## Function to get response from GPT-Neo model
def getGPTNeoResponse(input_text, no_words, blog_style):
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
    # Define the prompt template
    template = """
        Write a blog for {blog_style} on the topic: {input_text} 
        within {no_words} words.
        """
    
    # Format the prompt
    prompt = template.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the response
    output = model.generate(
        inputs.input_ids, 
        max_length=int(no_words) + len(inputs.input_ids[0]), 
        temperature=0.7,  # Adjusted for more natural responses
        num_return_sequences=1
    )
    
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)
    
    return response

st.set_page_config(
    page_title="Generate Blogs",
    page_icon='📝',
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Generate Blogs 📝")

input_text = st.text_area("Enter your blog prompt here:")

## Creating two more columns for 2 additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of words:')

with col2:
    blog_style = st.selectbox('Writing the blog for:', ['Researchers', 'Data Scientists', 'Common People'], index=0)

submit = st.button("Generate")

## Final Response
if submit:
    st.write(getGPTNeoResponse(input_text, no_words, blog_style))

