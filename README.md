# nl2sql
Use case based on LangChain in nl2sql demo
# nl2sql 整体架构
- 核心原则：借助CoT和ReAct Agent实现，LLM模型同时借助Student小模型和Remote Teacher大模型。对数据安全要求高、简单的子任务用Student模型实现
复杂且数据远端子任务用Teacher模型实现。
- 主要步骤：
    - 1. 原始查询语句正则、NLP模糊匹配，快速匹配SQL模板库；
    - 2. 匹配失败则 LLM模型解析原始查询Input，可用teacher模型完成语义理解、实体识别，较高准确度；
    - 3. 再次根据LLM解析结果，查询匹配SQL模板，未能成功匹配的走sql生成逻辑；
    - 4. 配置Prompt模板，检查SQL语句生成质量和安全性；
    - 5. 调用本地轻量级数据库运行SQL，检查正确性；
    - 6. 无法正确输出结果是，SQL chain启动ReAct机制，调用Student LLM 重新生成重复进行3~4步；
         失败超过2~3次，对SQL语句加密并调用Teacher LLM修改语句；
    - 7. 若SQL运行正确
    ![img.png](img.png)
