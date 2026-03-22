# 模块说明：
# - 旧 external RAG 子系统的导出层。
# - 主要给旧 demo 和旧验收链路复用。

# RAG 检索模块
from .semantic_retriever import SemanticRetriever

__all__ = ["SemanticRetriever"]
