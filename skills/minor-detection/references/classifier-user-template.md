# Classifier User Template

请基于下面的证据包完成一次未成年人识别。

输出要求：
- 严格遵循给定 schema。
- 如果证据不足，可以保守，但不要遗漏字段。
- `evidence` 中要把直接证据、证据总结、历史证据、检索证据、时间证据、冲突证据分开写。
- `evidence.direct_evidence` 只能写当前对话里可以直接定位到的引用式证据片段，尽量短，尽量接近原话。
- `evidence.direct_evidence` 只能摘自当前 `conversation`，不能引用或改写 `retrieved_cases`、时间特征、历史画像里的句子。
- 摘录时尽量逐字，不要为了读起来顺而补字改字，不要把原句换成“用户提到……”“这表明……”这类描述。
- 写完后请逐条核对：每条 `direct_evidence` 都必须能在 `conversation_text` 中直接找到；找不到就不要输出在 `direct_evidence` 中。
- 不要把“表明其学生身份”“进一步佐证”“符合未成年人的语言风格”等总结句写进 `direct_evidence`。
- 如果需要概括这些直接证据说明它们为什么支持结论，请写到 `evidence.evidence_summary`。

Schema:
{{OUTPUT_SCHEMA}}

Evidence Package:
{{EVIDENCE_PACKAGE_JSON}}
