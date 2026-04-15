import asyncio
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=4096, use_fp16=False)

nodes = [NodeWithScore(node=TextNode(text=f"This is node {i}"), score=0.5) for i in range(25)]

try:
    res = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="test"))
    print("Success. Elements:", len(res))
except Exception as e:
    import traceback
    traceback.print_exc()
