##############################################################
# RAG over PDF with FAISS & Streamlit
# 
# 필요 패키지 설치 커맨드
# 주의점 : 미니콘다 가상환경을 새로 만들어서 사용할 것, LangFlow 환경에 설치시 기존 개발 환경과 충돌 발생
# 추천 Python version : 3.10
# pip install uv
# uv pip install -U streamlit langchain langchain-community langchain-openai sentence-transformers faiss-cpu pypdf
# pip install -U langchain-huggingface
#
# 실행 방법 : # streamlit run rag_pdf.py
#
# 로컬 LLM 모델 사용시 올라마 설치 및 모델 다운로드 해놓을 것
# https://ollama.com/download
##############################################################

import os
import io
import tempfile
from typing import List

# OpenAI 사용시 API 키 직접 설정, "올라마"로 로컬모델 사용시는 셋팅안해도 됨.
os.environ['OPENAI_API_KEY'] = ''

# 스트림릿 라이브러리
import streamlit as st

# 파이선 로더
from langchain_community.document_loaders import PyPDFLoader

# 벡터디비 생성용 라이브러리
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM - OpenAI 또는 Ollama(로컬)
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# UI 설정
st.set_page_config(page_title="PDF RAG with FAISS", page_icon="📚", layout="wide")
# 이모지 사용하고 싶으면? https://emojipedia.org/ 에서 이모지 복사해서 붙여서 사용, 이모지는 컬러 폰트 Windows → Segoe UI Emoji
st.title("✨ 여러 PDF 를 로딩하고, 각각 질문해 봅니다.")

# UI 설정 - 사이드바
with st.sidebar:
    st.header("설정")

    provider = st.selectbox(
        "LLM 선택",
        [
            "Ollama (로컬)",
            "OpenAI (클라우드)",
        ],
        index=1,
    )

    model_name = st.text_input(
        "모델 이름",
        # 올라마 설치 및 ollama run exaone3.5:2.4b 등으로 모델 로딩이 선행되어야 함
        # exaone3.5:2.4b  엑사원 모델의 경우 LG AI Research 팀에서 개발한 것으로, 작은 용량으로 gtx1050 에서도 돌릴수 있는 경량 모델, GPU 4GB 정도 사용
        value="exaone3.5:2.4b" if provider.startswith("Ollama") else "gpt-4o",     # "gpt-4o-mini"
        help="Ollama는 로컬에 해당 모델이 pull 되어 있어야 합니다. OpenAI는 API Key 필요",
    )

    embed_model = st.text_input(
        "임베딩 모델 (Sentence-Transformers)",
        value="sentence-transformers/all-MiniLM-L6-v2",
        #value="BAAI/bge-large-en-v1.5",
        help="임베딩 모델에 따라서 gpu 사용, 차원증가로 비용증가, 품질차이가 발생",
    )

    chunk_size = st.slider("청크 크기", 300, 2000, 850, 50)
    chunk_overlap = st.slider("청크 오버랩", 0, 400, 90, 10)
    top_k = st.slider("검색 문서 수 (k)", 1, 20, 16, 1)   # 검색 문서 후보 수

    persist = st.checkbox("FAISS 인덱스 디스크 저장(.faiss_index)", value=True)

# UI 설정 - 
st.markdown(
    """
> 검색 증강 생성 - RAG (Retrieval-Augmented Generation) 원리 
>
> PDF 문서 Thunking -> FAISS Vector DB -> (질의 시점) DB Retriever 로 TOP_K 갯수만큼 유사한 Chunk Return -> 리턴된 Chunk 를 LLM 이 추론 
>
> Chunk 사이즈를 1000, Overlap 을 100, TOP_K 를 10 개로 변경 후 다시 검색해 보세요.
>
> 성능 좋은 임베딩 모델 및 추론모델 사용시 검색 품질이 좋아지나, GPU 및 하드웨어가 필요한 시점이 생깁니다.

**사용법**
1) 왼쪽에서 LLM/임베딩/청크 설정을 고릅니다.  
2) 아래 영역에 PDF 파일(복수 가능)을 업로드하고 **인덱스 생성**을 누릅니다.
3) 질문을 입력하면 RAG로 답변과 출처(페이지)를 반환합니다.
    """
)

# st.session_state : 브라우져 세션별 고유한 상태 값사용을 위한 딕셔너리
if "faiss" not in st.session_state:
    st.session_state.faiss = None   # Vector DB
if "docs_meta" not in st.session_state:
    st.session_state.docs_meta = [] # Vector DB Meta

# 파일 업로더 콘트롤 - BytesIO 스트림 타입
uploaded_files = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)

# 인덱스 생성 / 인덱스 초기화 콘트롤
col_build, col_reset = st.columns([1, 1])

# 리소스 캐싱 데코레이터, Streamlit 서버가 살아있는 동안 (앱 리로드 시 캐시 유지) 임베딩 모델을 초기로드시 한번만 로드하고, 재사용
@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    # HuggingFaceEmbeddings는 sentence-transformers 기반
    return HuggingFaceEmbeddings(model_name=model_name)

def load_pdfs_to_docs(files: List[io.BytesIO]):
    docs_all, meta_all = [], []

    for f in files:
        strFileName = getattr(f, "name", "uploaded.pdf")
        print("파일 처리중...", strFileName)

        # A) 업로더 포인터 초기화
        try:
            f.seek(0)
        except Exception:
            pass

        # 원본 이름으로 임시 저장(선택) — temp 경로 때문에 헷갈리면 권장
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"uploaded_{os.path.basename(strFileName)}")
        with open(tmp_path, "wb") as out:
            out.write(f.read())

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        cleaned_docs = []
        for d in docs:
            text = (d.page_content or "").strip()
            if not text:                 # B) 빈 텍스트 페이지 제거
                continue
            d.metadata["source"] = strFileName  # temp 경로 → 원본 파일명으로 고정
            pg = d.metadata.get("page", d.metadata.get("page_number", None))
            meta_all.append({
                "source": strFileName,
                "page": int(pg) + 1 if isinstance(pg, int) else pg,
                "content": text[:200].replace("\n", " ")
            })
            cleaned_docs.append(d)

        docs_all.extend(cleaned_docs)

    return docs_all, meta_all

# FAISS Vector Store 만들기
def build_faiss_index(docs, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    # RecursiveCharacterTextSplitter 을 통해 의미가 끊기지 않게 Chunk 를 잘라준다. separators 의 순서대로 Try
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    # RecursiveCharacterTextSplitter 을 사용해서 청크 만들기
    chunks = splitter.split_documents(docs)
    
    # 임베딩용 모델
    embeddings = get_embeddings(embedding_model_name)

    # 임베딩용 모델을 사용해서 FAISS 에서 Vector Store 를 생성
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    
    return vs

with col_build:
    if st.button("🔨 벡터 DB 생성", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드하세요.")
        else:
            with st.spinner("임베딩 & 인덱스 생성 중..."):
                # 파일 로드 - docs, meta 정보 
                docs, meta = load_pdfs_to_docs(uploaded_files)
                
                # 인덱스 빌드
                vs = build_faiss_index(docs, embed_model, chunk_size, chunk_overlap)
                
                # 벡터스토어
                st.session_state.faiss = vs
                # 메타정보
                st.session_state.docs_meta = meta
                
                # 디스크 저장
                if persist:
                    FAISS.save_local(vs, folder_path=".faiss_index")

            st.success("FAISS DB 생성 완료!")

with col_reset:
    if st.button("♻️ 벡터 DB 초기화", use_container_width=True):
        st.session_state.faiss = None
        st.session_state.docs_meta = []
        st.success("초기화되었습니다.")

# 기존 저장 인덱스 로드 버튼
col_load, col_dummy = st.columns([1, 3])

with col_load:
    if st.button("💾 저장된 인덱스 로드(.faiss_index)"):
        try:
            embeddings = get_embeddings(embed_model)

            vs = FAISS.load_local(
                folder_path=".faiss_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

            st.session_state.faiss = vs
            st.info("저장된 인덱스를 불러왔습니다.")
        except Exception as e:
            st.error(f"로드 실패: {e}")

st.divider()

st.markdown(
    """
    ### 예시 질문
    > 소나기의 주인공은?, 소녀는 누구인가?
    >
    > 경영지원 신입사원 JOB DESCRIPTION 을 알려주세요
    >
    > 에어컨에 물이 샙니다, 삼성 에어컨 서비스 전화번호 알려주세요
    """)

# 질의/응답
query = st.text_input("질문을 입력하세요", placeholder="예) 소나기의 주인공은?, 경영지원 신입사원 JOB DESCRIPTION 을 알려주세요, 에어컨에 물이 샙니다")

# LLM 준비
def get_llm():
    if provider.startswith("Ollama"):
        if ChatOllama is None:
            st.error("ChatOllama를 사용할 수 없습니다. 'langchain-community' 패키지와 Ollama가 필요합니다.")
            st.stop()
        return ChatOllama(model=model_name, temperature=0.1)
    else:
        if ChatOpenAI is None:
            st.error("OpenAI 패키지가 필요합니다: 'langchain-openai'")
            st.stop()
        # 환경 변수에 OPENAI_API_KEY가 있어야 함
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다. .env 또는 환경변수로 설정하세요.")
        return ChatOpenAI(model=model_name, temperature=0.1)

# RAG 용 프롬프트 설정
#
# RAG 시 중요 프롬프트 - 할루시네이션 방지 및 출처 출력
# "당신은 주어진 컨텍스트만을 사용하여 한국어로 간결하고 정확하게 답변하는 도우미입니다. "
# "컨텍스트에 없는 내용은 추측하지 말고 모른다고 말하세요."
def make_rag_prompt(question: str, context_chunks: List[str]):
    joined = "\n\n".join([f"[컨텍스트 {i+1}]\n" + c for i, c in enumerate(context_chunks)])
    
    sys = (
        "당신은 주어진 컨텍스트만을 사용하여 한국어로 간결하고 정확하게 답변하는 도우미입니다. "
        "컨텍스트에 없는 내용은 추측하지 말고 모른다고 말하세요."
    )
    user = f"질문: {question}\n\n컨텍스트:\n{joined}\n\n답변:"

    return sys, user

if query:
    if st.session_state.faiss is None:
        st.warning("먼저 PDF를 업로드하고 인덱스를 생성(또는 로드)하세요.")
    else:
        # 검색
        retriever = st.session_state.faiss.as_retriever(search_kwargs={"k": top_k})
        
        docs = retriever.invoke(query)
        
        if not docs:  # C) 0건 가드
            st.warning("검색 결과가 없습니다. 질문을 구체화하거나 청크/임베딩 설정을 조정하세요.")
            st.stop()

        # 컨텍스트 준비 (길이 제한)
        contexts: List[str] = []
        sources = []
        for d in docs:
            text = d.page_content
            meta = d.metadata or {}
            page = meta.get("page", meta.get("page_number", None))
            if isinstance(page, int):
                page = page + 1
            src = meta.get("source", "PDF")
            # 짧은 스니펫
            snippet = (text[:300] + "...") if len(text) > 300 else text
            contexts.append(snippet)
            sources.append({"source": os.path.basename(src), "page": page, "snippet": snippet})

        llm = get_llm()
        system_prompt, user_prompt = make_rag_prompt(query, contexts)

        with st.spinner("LLM 응답 생성 중..."):
            # Chat API 스타일 호출
            response = llm.invoke([
                ("system", system_prompt),
                ("user", user_prompt),
            ])
            answer = response.content if hasattr(response, "content") else str(response)

        # 출력
        st.subheader("🔎 답변")
        st.write(answer)

        with st.expander("📌 출처 (Top-k 문서) 보기", expanded=False):
            for i, s in enumerate(sources, start=1):
                st.markdown(f"**{i}. {s['source']}** — p.{s['page']}\n\n> {s['snippet']}")

