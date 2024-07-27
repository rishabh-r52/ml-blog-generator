import streamlit as st
from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers


def getllamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    template = f"""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=['style', 'text', 'n_words'], template=template)

    response = llm(prompt.format(style=blog_style, text=input_text, n_words=no_words))
    print(response)
    return response


st.set_page_config(
    page_title="Blog Generation App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the blog topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of words: ")

with col2:
    blog_style = st.selectbox("Writing the blog for: ", ("Researchers", "Data Scientists", "Common People"), index=0)

submit = st.button("Generate")

# Final Response
if submit:
    st.write(getllamaresponse(input_text, no_words, blog_style))
