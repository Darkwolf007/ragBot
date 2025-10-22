# ui_app.py
import streamlit as st
from query import search_query

st.set_page_config(page_title="TN Building Rules Search", layout="wide")

st.title("Tamil Nadu Combined Building Rules — Query")

query = st.text_input("Enter your question:")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = search_query(query, k=top_k)

    st.markdown("---")
    for r in results:
        st.subheader(f"#{r['rank']}  —  Page {r['page']}  |  Type: {r['type']}  |  Score: {r['score']:.3f}")
        if r["type"] == "text":
            st.write(r["content"])
        elif r["type"] == "image":
            st.image(r["path"], caption=f"Page {r['page']}")
            if r.get("ocr_text"):
                st.caption("📝 OCR: " + r["ocr_text"])
        st.markdown("---")
