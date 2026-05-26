# src/evaluation.py
import asyncio
from typing import List, Optional, Any
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class ChatGroqWithN(BaseChatModel):
    """Composition wrapper targeting ChatGroq to consistently intercept and scale N generations."""

    # We must explicitly use keyword arguments or declare a placeholder initialization
    # to maintain compatibility with BaseChatModel's internal Pydantic configuration.
    def __init__(self, model: str, temperature: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        # Prefixed with an underscore to tell Pydantic: "This is a private property, don't validate it"
        self.__dict__["_underlying_client"] = ChatGroq(
            model=model, temperature=temperature)
        self.__dict__["_model_name"] = model

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
        requested_n = kwargs.pop("n", 1)
        generations = []

        # Execute sequentially to completely satisfy Ragas multi-turn samples without API drops
        for _ in range(requested_n):
            res = self._underlying_client.invoke(messages, stop=stop, **kwargs)
            generations.append(ChatGeneration(message=res))

        return ChatResult(generations=generations)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
        requested_n = kwargs.pop("n", 1)

        # Fan out concurrent tasks asynchronously for performant scoring loops
        tasks = [
            self._underlying_client.ainvoke(messages, stop=stop, **kwargs)
            for _ in range(requested_n)
        ]
        results = await asyncio.gather(*tasks)

        generations = [ChatGeneration(message=res) for res in results]
        return ChatResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq-mcp-eval-wrapper"


def assess_agent_health(query: str, agent_response: str, code_contexts: list) -> dict:
    """Evaluates the alignment and quality of agent answers using modern Ragas metrics."""
    groq_llm = ChatGroqWithN(model="llama-3.3-70b-versatile", temperature=0.0)
    eval_llm = LangchainLLMWrapper(groq_llm)

    base_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5")
    eval_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    faithfulness_metric = Faithfulness(llm=eval_llm)
    relevance_metric = ResponseRelevancy(
        llm=eval_llm, embeddings=eval_embeddings)

    payload = SingleTurnSample(
        user_input=query,
        response=agent_response,
        retrieved_contexts=[str(c.get("source_code", ""))[:1500]
                            for c in code_contexts]
    )

    dataset = EvaluationDataset(samples=[payload])

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, relevance_metric],
            llm=eval_llm
        )

        results_df = results.to_pandas()
        # Clean check for metrics keys
        faithfulness_score = results_df["faithfulness"].iloc[0] if "faithfulness" in results_df.columns else 0.0
        response_relevance_score = results_df["user_input_response_relevancy"].iloc[0] if "user_input_response_relevancy" in results_df.columns else (
            results_df["answer_relevancy"].iloc[0] if "answer_relevancy" in results_df.columns else 0.0)

        return {
            'faithfulness_score': faithfulness_score,
            'response_relevance_score': response_relevance_score,
            'status': "PASS" if (faithfulness_score >= 0.70 and response_relevance_score >= 0.70) else "FAIL"
        }

    except Exception as e:
        print(f"❌ Ragas evaluation execution failed: {str(e)}")
        return {"faithfulness_score": 0.0, "response_relevance_score": 0.0, "status": "ERROR"}
