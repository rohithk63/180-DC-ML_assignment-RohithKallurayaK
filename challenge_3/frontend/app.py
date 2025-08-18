import streamlit as st
import requests

st.title("Courtroom Clash: AI Lawyers Battle It Out! 🤖⚖️🔥")

if st.button("Generate Random Case"):
    res = requests.post("http://127.0.0.1:8000/generate_case")
    case = res.json()["case"]
    st.session_state['case'] = case

if 'case' in st.session_state:
    st.subheader("Case")
    st.write(st.session_state['case'])

    if st.button("Start Debate"):
        res = requests.post("http://127.0.0.1:8000/debate", json={"case_text": st.session_state['case']})
        debate = res.json()
        st.subheader("RAG Lawyer Argument")
        st.write(debate['rag_lawyer']['argument'])
        st.write("Citations:", debate['rag_lawyer']['citations'])

        st.subheader("Chaos Lawyer Argument")
        st.write(debate['chaos_lawyer']['argument'])

    winner = st.radio("Judge Decision: Who wins?", ["RAG Lawyer", "Chaos Lawyer"])
    if st.button("Submit Verdict"):
        res = requests.post("http://127.0.0.1:8000/judge_decision", json={"winner": winner})
        st.write(res.json()["verdict"])
