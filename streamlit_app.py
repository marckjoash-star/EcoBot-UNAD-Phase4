"""EcoBot Hybrid Streamlit interface: lightweight retrieval + remote open LLM."""
import os
import streamlit as st
from ecobot_hybrid_core import Memory, chat, DEFAULT_MODEL

st.set_page_config(page_title='EcoBot Hybrid | UNAD',page_icon='🌿',layout='centered')
st.title('🌿 EcoBot')
st.caption('Environmental education | Coastal ecosystem conservation | Hybrid documentary retrieval + open LLM')
with st.sidebar:
    st.header('About EcoBot')
    st.write('Ask about mangrove species, blue carbon, coral reefs and plastic pollution. Español / English.')
    st.warning('Educational content only. No live environmental monitoring or professional site assessments.')
    st.caption('Sources: curated educational paraphrases of NOAA and Florida DEP documents. Always check original pages for important decisions.')
    if st.button('Clear conversation',use_container_width=True):
        st.session_state.history=[]
        st.session_state.memory=Memory()
        st.rerun()
    st.divider()
    st.caption('UNAD · Natural Language Processing · Group 203238430_8')
    st.caption('Lilia Maria Ward Bowie · Phase 4')

if 'history' not in st.session_state: st.session_state.history=[]
if 'memory' not in st.session_state: st.session_state.memory=Memory()

for m in st.session_state.history:
    with st.chat_message(m['role']):
        st.markdown(m['text'])
        if m.get('sources'):
            with st.expander('Documentary sources'):
                for index,d in enumerate(m['sources'],1):
                    st.markdown(f'[{index}. {d["title"]}]({d["url"]})')
        if m.get('status'): st.caption('Response method: '+m['status'])

if not st.session_state.history:
    st.markdown('**Try:**')
    st.caption('• ¿Qué especies de manglar existen en el Caribe?')
    st.caption('• ¿Cuáles son sus diferencias?')
    st.caption('• What causes coral bleaching?')

question=st.chat_input('Ask an environmental question / Haz una pregunta')
if question:
    st.session_state.history.append({'role':'user','text':question})
    with st.chat_message('user'): st.markdown(question)
    try:
        hf_token=st.secrets.get('HF_TOKEN','') or os.getenv('HF_TOKEN','')
        hf_model=st.secrets.get('HF_MODEL',DEFAULT_MODEL) or DEFAULT_MODEL
    except Exception:
        hf_token=os.getenv('HF_TOKEN','')
        hf_model=os.getenv('HF_MODEL',DEFAULT_MODEL)
    with st.chat_message('assistant'):
        with st.spinner('Retrieving evidence and generating a response...'):
            result=chat(question,st.session_state.memory,token=hf_token,model=hf_model)
        st.markdown(result['answer'])
        if result['sources']:
            with st.expander('Documentary sources'):
                for index,d in enumerate(result['sources'],1):
                    st.markdown(f'[{index}. {d["title"]}]({d["url"]})')
        st.caption('Response method: '+result['status'])
    st.session_state.history.append({'role':'assistant','text':result['answer'],'sources':result['sources'],'status':result['status']})
