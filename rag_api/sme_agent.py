import os
import json
import ast
from typing import List, Dict, Tuple, Optional

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.agents import AgentExecutor, Tool, create_react_agent
from langchain_classic import hub
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from reportlab.lib.pagesizes import letter
import markdown
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

class MDToolInput(BaseModel):
    """Input schema for the SaveAsMDTool."""
    content: str = Field(description="The text content to be written to the Markdown file.")
    filename: str = Field(description="The desired name for the output Markdown file, e.g., 'notes.md'.")

class SaveAsMDTool(BaseTool):
    """A tool that saves given text content into a Markdown (.md) file."""
    name: str = "Save As Markdown"
    description: str = "Use this tool to save text content into a .md file. It takes content and a filename as input."
    args_schema: Type[BaseModel] = MDToolInput

    def _run(self, content: str, filename: str) -> str:
        """Saves the content to a Markdown file and returns the status."""
        try:
            output_dir = "generated_documents"
            os.makedirs(output_dir, exist_ok=True)
            if not filename.endswith(".md"):
                filename += ".md"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully saved content to {file_path}"
        except Exception as e:
            return f"Error saving Markdown file: {e}"


class PDFToolInput(BaseModel):
    """Input schema for the SaveAsPDFTool."""
    content: str = Field(description="The text content to be written to the PDF.")
    filename: str = Field(description="The desired name for the output PDF file, e.g., 'assignment.pdf'.")

class SaveAsPDFTool(BaseTool):
    """A tool that saves given text content into a PDF file."""
    name: str = "Save As PDF"
    description: str = "Use this tool to save text content into a .pdf file. It takes content and a filename as input."
    args_schema: Type[BaseModel] = PDFToolInput

    def _run(self, content: str, filename: str) -> str:
        """Saves the content to a PDF and returns the status."""
        try:
            output_dir = "generated_documents"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            html_content = markdown.markdown(content)
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            for line in html_content.splitlines():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 12))
            doc.build(story)
            return f"Successfully saved Markdown content to {file_path}"
        except Exception as e:
            return f"Error saving PDF: {e}"

class SMEAgent:
    """
    A Subject Matter Expert agent that uses an advanced
    Parent-Child retrieval pipeline with reranking.
    """
    def __init__(self, assets_path: str = "assets.json"):
        print("Initializing SMEAgent...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.parent_dir = os.path.dirname(script_dir) 
            self._load_credentials(os.path.join(self.parent_dir, assets_path))
        except NameError:
            self.parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            self._load_credentials(os.path.join(self.parent_dir, assets_path))
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=self.GEMINI_API_KEY, temperature=0)
        
        print("Loading embedding and reranker models...")
        self.embeddings_model = SentenceTransformer('all-mpnet-base-v2')
        self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
        print("Models loaded.")

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.child_index = self.pc.Index("sme-agent-child-chunks")
        self.parent_index = self.pc.Index("sme-agent-parent-chunks")
        print("Pinecone indexes connected.")

        tools = self._create_tools()
        
        agent_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        
        print("SMEAgent initialized successfully.")

    def _load_credentials(self, file_path: str):
        self.GEMINI_API_KEY = self._get_api_key("gemini_api_key", file_path)
        os.environ["PINECONE_API_KEY"] = self._get_api_key("pinecone_api_key", file_path)

        if not all([os.environ["PINECONE_API_KEY"], self.GEMINI_API_KEY]):
            raise ValueError("All API keys must be set in the assets file.")

    def _get_api_key(self, key_name: str, file_path: str):
        try:
            with open(file_path, 'r') as f:
                credentials = json.load(f)
            return credentials.get(key_name)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading credentials file at {file_path}: {e}")
            return None

    def _create_tools(self) -> list:
        """
        Creates the tools for the agent, replacing RetrievalQA with
        our advanced RAG pipeline.
        """
        return [
            Tool(
                name="Question Answering",
                func=self._run_qa_rag, 
                description="Use this tool ONLY to answer specific questions about data science concepts. Input should be a clear question."
            ),
            Tool(
                name="Assignment Content Generator",
                func=self._run_assignment_rag, 
                description="Use this tool to generate the TEXT CONTENT for an assignment, quiz, or study guide. This tool ONLY generates text; it does not save files."
            ),
            Tool(
                name="Save As PDF",
                func=lambda tool_input_str: SaveAsPDFTool().run(ast.literal_eval(tool_input_str)),
                description="Use this tool to save text content into a .pdf file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
            ),
            Tool(
                name="Save As Markdown",
                func=lambda tool_input_str: SaveAsMDTool().run(ast.literal_eval(tool_input_str)),
                description="Use this tool to save text content into a .md file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
            )
        ]

    def _retrieve_and_rerank(self, query: str, final_top_k: int = 3) -> str:
        """
        Performs the full parent-child retrieval and reranking pipeline.
        This is the core of our new RAG logic.
        """
        print(f"RAG: Retrieving context for query: '{query}'")
        query_embedding = self.embeddings_model.encode(query).tolist()
        retrieval_top_k = 20  
        child_results = self.child_index.query(
            vector=query_embedding,
            top_k=retrieval_top_k,
            include_metadata=True
        )
        parent_ids = []
        for match in child_results.get('matches', []):
            if 'parent_chunk_id' in match.get('metadata', {}):
                parent_ids.append(match['metadata']['parent_chunk_id'])
        parent_ids = list(set(parent_ids))
        if not parent_ids:
            print("RAG: No parent documents found from retrieval.")
            return "No relevant context found."
        print(f"RAG: Fetching {len(parent_ids)} unique parent documents...")
        parent_results = self.parent_index.fetch(ids=parent_ids)        
        docs_to_process = []
        for parent_id, doc_data in parent_results['vectors'].items():
            docs_to_process.append({
                "id": parent_id,
                "text": doc_data['metadata'].get('text', ''),
                "source": doc_data['metadata'].get('source', 'Unknown')
            })
        try:
            print(f"RAG: Reranking {len(docs_to_process)} candidate documents...")
            pairs = [[query, doc['text']] for doc in docs_to_process]
            scores = self.reranker.predict(pairs, show_progress_bar=False)            
            reranked_docs = list(zip(scores, docs_to_process))
            reranked_docs.sort(key=lambda x: x[0], reverse=True)            
            context_chunks = []
            print(f"RAG: Building context from top {final_top_k} reranked results.")
            for score, doc in reranked_docs[:final_top_k]:
                context_chunks.append(doc['text'])
            
            return "\n\n---\n\n".join(context_chunks)

        except Exception as e:
            print(f"RAG: Reranking failed ({e}). Falling back to basic similarity.")
            # Fallback: Just use the top k documents from the fetch
            fallback_context = [doc['text'] for doc in docs_to_process[:final_top_k]]
            return "\n\n---\n\n".join(fallback_context)

    def _get_qa_prompt(self):
        template = """You are a helpful Data Science assistant.
Read the user's QUESTION and the provided CONTEXT carefully.
Think step-by-step to find the relevant information in the context that answers the question.

Finally, answer the QUESTION based ONLY on the CONTEXT and answer step-by-step.
If the context does not contain the answer, state that you cannot answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        return ChatPromptTemplate.from_template(template)

    def _get_assignment_prompt(self):
        template = """You are a Data Science instructor. Create the text content for an assignment based on the user's request, using the provided context.
Generate only the text for the assignment.

CONTEXT:
{context}

USER'S REQUEST:
{question}

ASSIGNMENT CONTENT:"""
        return ChatPromptTemplate.from_template(template)

    def _run_qa_rag(self, query: str) -> str:
        """
        Runs the full RAG-Rerank-Generate chain for Q&A.
        """
        print("--- Running QA RAG Chain ---")
        qa_prompt = self._get_qa_prompt()
        
        # Build the LCEL chain
        rag_chain = (
            {"context": self._retrieve_and_rerank, "question": RunnablePassthrough()}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)
    
    def _run_assignment_rag(self, query: str) -> str:
        """
        Runs the full RAG-Rerank-Generate chain for Assignment Generation.
        """
        print("--- Running Assignment RAG Chain ---")
        assignment_prompt = self._get_assignment_prompt()
        
        # Build the LCEL chain
        rag_chain = (
            {"context": self._retrieve_and_rerank, "question": RunnablePassthrough()}
            | assignment_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)

    def run(self, user_input: str) -> dict:
        print(f"Agent is processing input: '{user_input}'")
        return self.agent_executor.invoke({"input": user_input})