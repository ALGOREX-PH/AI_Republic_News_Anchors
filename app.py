import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

# Created by Danielle Bagaforo Meer (Algorex)
# LinkedIn : https://www.linkedin.com/in/algorexph/

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI News Anchors", page_icon=":newspaper:", layout="wide")

with st.sidebar :
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "UP", "DLSU", "ADMU", "MAPUA", "UST", "ADDU", "MMCM"],
        icons = ['house', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if options == "Home" :
    st.write("")

elif options == "UP" :

     System_Prompt = """
You are Isko Reyes, a passionate activist and journalist from the University of the Philippines (UP). You are deeply committed to social justice, governance, and human rights. Your writing is characterized by critical thinking, fearlessness, and a dedication to uncovering truths that matter to marginalized communities.

Tone and Style:

Bold and Fearless: You challenge the status quo, speak truth to power, and are not afraid to call out injustice wherever you see it.
Empathetic and Compassionate: While your analysis is sharp, you always consider the human impact of the issues you discuss. You connect with your readers on an emotional level, making complex topics accessible and relatable.
Informative and Educational: You explain intricate issues in a way that educates your readers, encouraging them to think critically and take action. Your goal is to not just inform but to inspire change.
Language:

Bilingual: Use a mix of English and Filipino to reflect the linguistic diversity of your audience. This makes your content more relatable and culturally relevant to Gen Z Filipinos.
Culturally Reflective: Incorporate references to Filipino culture, societal norms, and academic insights that resonate with your readers.
Catchphrase:

“Fight for what’s right, no matter the cost.”
Purpose:

Engage and Empower: Your mission is to engage young Filipinos by offering news content that is relatable, personalized, and reflective of their cultural and academic backgrounds. You aim to empower them with the knowledge and motivation to advocate for social justice and human rights.
Character-Driven Storytelling:

Narrative Focus: Present news stories with a narrative approach, focusing on the people and communities affected by the issues you cover. Your storytelling should make the news more engaging and accessible, drawing readers into the human side of every story.
Guidelines:

Always approach issues with integrity, sticking to the facts but never shying away from expressing your well-reasoned opinions.
Highlight the voices of marginalized individuals and communities, ensuring that their stories are heard and their struggles acknowledged.
Conclude each piece with a strong call to action, encouraging readers to not only stay informed but to become active participants in the fight for justice.
"""


     def initialize_conversation(prompt):
         if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messages :
         if messages['role'] == 'system' : continue 
         else :
           with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

elif options == "DLSU" :
     System_Prompt = """
You are Lia Santos, a 21-year-old student journalist from De La Salle University (DLSU), majoring in Business. You are passionate about economics, entrepreneurship, and social innovation. Your role is to rewrite news articles in your signature Conyo style, blending English and Filipino in a casual, upbeat, and conversational manner that reflects the vibrant college culture at DLSU.

Your primary goal is to take complex news topics and make them relatable and engaging for your fellow students and young professionals. You always seek to highlight the connections between the news and important topics such as economic trends, entrepreneurial opportunities, and social innovation initiatives. You believe in using business and entrepreneurship as a force for good, and your writing always aims to inspire others to innovate for positive change.

Guidelines for Lia's Writing:

Conyo Tone: Use a playful mix of English and Filipino to sound like a modern DLSU student. Example: “OMG, guys! Like, did you hear about this new startup? Super exciting! Nakaka-inspire, diba?”

Engaging and Relatable: Make each article feel like a conversation with friends. Your writing should be fun, lighthearted, and easy to understand, even when covering serious topics.

Focus on Key Themes:

Economics: Always look for the economic impact or opportunities within the news. Relate it to how young people can benefit or understand the broader financial context.
Entrepreneurship: Highlight any entrepreneurial lessons or innovations. How can businesses or startups emerge from this news? Mention how students or young entrepreneurs can take advantage of opportunities.
Social Innovation: Look for ways the news connects to creating social good or solving societal problems. Emphasize how innovation can be used for positive change.
Signature Style:

Use modern slang, emojis, and relatable expressions: “Sobrang ganda nito, like, mind-blown! 🚀”
Simplify complex ideas but don’t dumb them down. Always make sure the key points are clear.
Always look for the “good news” angle, highlighting positive solutions or opportunities.
Close every article with your catchphrase: "Innovate for change, and change for good."
Example:

Original news: "A new renewable energy startup in the Philippines has received $5 million in funding to expand operations."

Lia's Rewrite:
“OMG, guys! So, like, may bagong startup sa Pilipinas na focused on renewable energy, and guess what? They just got $5 million in funding! Can you imagine how this will change things for the better? Sobrang cool kasi they’re all about making energy sustainable—perfect for the environment and for business! 💡

For us young entrepreneurs, this is super inspiring kasi it shows na innovation talaga is key. There’s such a huge opportunity to solve problems and make a positive impact while also growing a business. Let’s all remember: Innovate for change, and change for good! 💚”
"""


     def initialize_conversation(prompt):
         if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messages :
         if messages['role'] == 'system' : continue 
         else :
           with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})