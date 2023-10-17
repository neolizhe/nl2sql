# coding : utf - 8
import os
import openai
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain

openai.api_type = "azure"
openai.api_base = "https://camc-datacenter-test.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_KEY"] = "28eb23f1e84a4ec2842c5dbbdbb95184"
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    db = SQLDatabase.from_uri("sqlite:///database/Chinook.db")
    llm = OpenAI(engine="test-deploy",
                       model_name="text-davinci-003",
                       temperature=0.2)

    # llm = openai.ChatCompletion.create(
    #         engine="test-deploy",
    #         messages=dialogue_history,
    #         temperature=0.7,
    #         max_tokens=800,
    #         top_p=0.95,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop=None)
    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)
    from chain.CustomSQLChain import CustomSQLChain

    db_chain = CustomSQLChain.from_multi_llm(student_llm=llm, teacher_llm=llm,
                                             valid_db=db, verbose=True, return_intermediate_steps=True)
    db_chain.run("专辑销量第5高但是歌曲数量第10高的基金经理是一个人吗？如果不是的话分别列出两人的名字。")
    # res = db_chain.apply([{"query":"2023年电车销量最高的是哪个"}])
    # print(res[0]["intermediate_steps"])
#
