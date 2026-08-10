import streamlit as st

from rag import generate_answer

st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

query = st.text_input(
    "Ask a question from your PDFs"
)

if st.button("Search"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents..."
        ):

            answer, citations, docs = (
                generate_answer(query)
            )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for c in citations:

            st.write("•", c)

        st.subheader(
            "Retrieved Chunks"
        )

        for i, doc in enumerate(docs):

            with st.expander(
                f"Chunk {i+1}"
            ):

                st.write(doc)