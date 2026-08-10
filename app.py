import streamlit as st
from rag import generate_answer

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

st.write(
    "Ask questions about your uploaded PDF documents."
)

query = st.text_input(
    "Ask a question",
    placeholder="Example: What is machine learning?"
)

if st.button("🔍 Search", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching your PDF documents..."
        ):

            try:

                answer, citations, documents = (
                    generate_answer(query)
                )

                st.subheader("💡 Answer")

                st.write(answer)

                st.subheader("📄 Sources")

                if citations:

                    for citation in citations:
                        st.write(f"• {citation}")

                else:

                    st.info(
                        "No relevant PDF source found."
                    )

                if documents:

                    with st.expander(
                        "🔎 Retrieved PDF Content"
                    ):

                        for i, document in enumerate(
                            documents
                        ):

                            st.markdown(
                                f"**Chunk {i + 1}**"
                            )

                            st.write(document)

            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing your question."
                )

                st.exception(e)


                