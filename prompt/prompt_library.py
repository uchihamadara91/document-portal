from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
        You are a highly capable assistant trained to analyze and summarize documents.
        Return only valid JSON matching the exact schema below.
        
        {format_instructions}

        Analyze this document:
        {document_texts}
        """)
