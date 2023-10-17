# coding : utf - 8
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers.enum import Enum, EnumOutputParser

_sqlite_prompt_retry = """You are a SQLite expert. Please strictly adhere to the following instructions: \
Step 1. Identify and correct the cause of the failure based on the provided SQLite dialect and runtime error, to ensure that the fixed SQLite dialect can be executed successfully. 
SQLite dialect is provided as: '{{last_dialect}}', Runtime error is provided as: '{{runtime_error}}', All useful tables schema: '{{table_info}}'. \
Step 2. Given the original query input, verify whether the corrected SQLite dialect is consistent with the original input '{{input}}',
If it is consistent, please output the corrected SQLite dialect and answer YES, otherwise answer NO and do not output SQLite dialect '{{format_instructions}}' 
"""



# Structured output parser
class AnswerEnum(Enum):
    YES = "yes"
    NO = "No"


response_schemas = [
    ResponseSchema(name="answer", type=AnswerEnum.__name__, description="According to the check result of input SQLite dialect, "
                                                               "Answer 'yes' or 'no'. 'yes' stands for corrected SQLite is aligned to "
                                                               "origin query intention, 'no' stands for the other side"),
    ResponseSchema(name="corrected_sqlite_dialect",
                   description="Corrected SQLite dialect that aligned to query intention should be put here")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)

SQLITE_PROMPT_RETRY = PromptTemplate.from_template(
    template=_sqlite_prompt_retry,
    template_format="jinja2"
)

if __name__ == "__main__":
    params = {
        "input":"Query how many cars in China",
        "table_info":"aaa",
        "last_dialect":"SELECT * FRO aaa",
        "runtime_error":"err",
        "format_instructions":output_parser.get_format_instructions()
    }
    print(SQLITE_PROMPT_RETRY.format_prompt(**params))
