# coding : utf - 8
# Inherit from SQLDAtabaseChain
import warnings
import re
from typing import Any, Dict, List, Optional

from langchain import LLMChain, SQLDatabase, BasePromptTemplate, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain_experimental.pydantic_v1 import Extra, Field, root_validator
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
# Prompt

from prompts.query_check_prompt import QUERY_CHECKER
from prompts.regenerate_sql_cmd import SQLITE_PROMPT_RETRY, output_parser

INTERMEDIATE_STEPS_KEY = "intermediate_steps"


class CustomSQLChain(SQLDatabaseChain):
    student_llm_chain: LLMChain = None
    teacher_llm_chain: Optional[LLMChain] = None
    """[Deprecated] LLM wrapper to use."""
    validate_database: SQLDatabase = Field()
    """SQL Database to connect to validate."""
    execute_database: SQLDatabase = Field()
    """SQL DB to exe at last."""
    prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 5
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_sql: bool = False
    """Will return sql-command directly without executing it"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""
    # use_query_checker: bool = False
    """Must the query checker tool should be used to attempt
    to fix the initial SQL from the LLM."""
    query_checker_prompt: Optional[BasePromptTemplate] = None
    """The prompt template that should be used by the query checker"""
    run_manager: Optional[CallbackManagerForChainRun] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None) -> \
            Dict[str, Any]:
        raise NotImplementedError("Async call not supported for this chain type.")

    def raise_deprecation(cls, values: Dict) -> Dict:
        # Drop LLMChain and use multi llm to generate LLMChain
        return values

    def sql_cmd_check(self, sql_cmd):
        # Check SQL command on syntax error , not care about aligned to query intention
        sql_cmd = sql_cmd + ';' if sql_cmd[-1] != ';' else sql_cmd
        print("="*50+"Origin sql"+"="*50)
        print(sql_cmd)
        query_checker_prompt = self.query_checker_prompt or PromptTemplate(
            template=QUERY_CHECKER, input_variables=["query", "dialect"]
        )

        # Ensure temperature of check LLM is zero
        temperature = self.student_llm_chain.llm.temperature
        self.student_llm_chain.llm.__setattr__(name="temperature", value=0.0)
        query_checker_chain = LLMChain(
            llm=self.student_llm_chain.llm, prompt=query_checker_prompt
        )
        query_checker_inputs = {
            "query": sql_cmd,
            "dialect": self.validate_database.dialect,
        }
        checked_sql_command: str = query_checker_chain.predict(
            callbacks=self.run_manager.get_child(), **query_checker_inputs
        ).strip()

        checked_sql_command = checked_sql_command + ';' if checked_sql_command[-1] != ';' else checked_sql_command
        print("=" * 50 + "Checked sql" + "=" * 50)
        print(checked_sql_command)
        match_s, match_e = re.search("SELECT.*?;", checked_sql_command).span()
        # Display checked sql in green color
        self.run_manager.on_text(
            checked_sql_command[match_s:match_e], color="green", verbose=self.verbose
        )

        # Reset LLM temperature
        self.student_llm_chain.llm.__setattr__(name="temperature", value=temperature)
        return checked_sql_command[match_s:match_e]

    def sql_cmd_execute(self, sql_cmd, is_fast_check=True):
        result_dict = {
            "is_succeed": True,
            "result": "",
            "runtime_info": ""
        }
        # Whether valid or run on real database
        if is_fast_check:
            result = self.validate_database.run_no_throw(sql_cmd)
            print(result)
            if result.startswith("Error"):
                result_dict["is_succeed"] = False
                result_dict["runtime_info"] = result[5:]
            else:
                result_dict["result"] = result
        else:
            result = self.execute_database.run_no_throw(sql_cmd)
            if result.startswith("Error"):
                result_dict["is_succeed"] = False
                result_dict["runtime_info"] = result[5:]
            else:
                result_dict["result"] = result
        return result_dict

    def reasoning_and_act_process(self, intermediate_steps, llm_inputs):
        MAX_TRIALS = 3
        reasoning_index = 0
        final_best_dialect = ""
        while True:
            if reasoning_index > MAX_TRIALS:
                break
            # 2. Use LLM chain to generate SQL cmd
            sql_cmd = self.student_llm_chain.predict(
                callbacks=self.run_manager.get_child(),
                **llm_inputs,
            ).strip()
            # 3. Check SQL raw cmd
            checked_sql_cmd = self.sql_cmd_check(sql_cmd)

            # 4. Run sql cmd in valid mode or exe mode
            valid_result = self.sql_cmd_execute(checked_sql_cmd, is_fast_check=True)

            if valid_result.get("is_succeed"):
                final_best_dialect = checked_sql_cmd
                print("ReAct chain ended at %s times" % reasoning_index)
                break
            else:
                runtime_error = valid_result.get("runtime_info")
                # retry prompt
                prompt_params = {
                    "input": llm_inputs.get("input"),
                    "table_info": llm_inputs.get("table_info"),
                    "last_dialect": checked_sql_cmd,
                    "runtime_error": runtime_error,
                    "format_instructions": output_parser.get_format_instructions()
                }
                retry_llm_chain = LLMChain(llm=self.teacher_llm_chain.llm,
                                           prompt=SQLITE_PROMPT_RETRY)
                check_result = retry_llm_chain.predict(callbacks=self.run_manager.get_child(),
                                                            **prompt_params).strip()
                print("~"*100 + check_result)
                # print(check_result)

                retry_output = output_parser.parse(check_result)
                if "answer" in retry_output.keys():
                    value = retry_output.get("answer")
                    if value.lower() == "yes" and "corrected_sqlite_dialect" in retry_output.keys():
                        vr = self.sql_cmd_execute(retry_output.get("corrected_sqlite_dialect"), is_fast_check=True)
                        if vr.get("is_succeed"):
                            final_best_dialect = retry_output.get("corrected_sqlite_dialect")
                            break
            reasoning_index += 1
            print("Reasoning Index %s" % reasoning_index)
        return final_best_dialect

    def parser_result(self, intermediate_steps, llm_inputs, sql_cmd, sql_result):
        llm_inputs["input"] += f"{{sql_cmd}}\nSQLResult: {{result}}\nAnswer:".format(sql_cmd=sql_cmd,result=sql_result)
        intermediate_steps.append(llm_inputs)

        # input: final answer
        final_result = self.student_llm_chain.predict(
            callbacks=self.run_manager.get_child(),
            **llm_inputs,
        ).strip()
        intermediate_steps.append(final_result)  # output: final answer
        self.run_manager.on_text(final_result, color="green", verbose=self.verbose)

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
        return chain_result

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        self.run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        self.run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.validate_database.get_table_info(table_names=table_names_to_use)

        # 1. Assemble LLM input to generate SQL cmd
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.validate_database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)

            # 4.ReAct Mechanism to get right SQL query result
            final_sql_cmd = self.reasoning_and_act_process(intermediate_steps, llm_inputs)

            # Use Remote Database to get final result
            result = self.sql_cmd_execute(final_sql_cmd, is_fast_check=False)

            self.run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            self.run_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                self.run_manager.on_text("\nAnswer:", verbose=self.verbose)
                final_result = self.parser_result(intermediate_steps=intermediate_steps,
                                                  llm_inputs=llm_inputs,
                                                  sql_cmd=final_sql_cmd,
                                                  sql_result=result)
            return final_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc

    @property
    def _chain_type(self) -> str:
        return "custom_sql_chain"

    @property
    def _run_output_key(self) -> str:
        if len(self.output_keys) < 1:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"one output key."
            )
        return self.output_keys[0]

    @classmethod
    def from_multi_llm(
            cls,
            student_llm: BaseLanguageModel,
            teacher_llm: Optional[BaseLanguageModel],
            valid_db: SQLDatabase,
            remote_db: Optional[SQLDatabase] = None,
            **kwargs: Any,
    ) -> SQLDatabaseChain:
        if not teacher_llm:
            teacher_llm = student_llm
        if not remote_db:
            remote_db = valid_db
        student_llm_chain = LLMChain(llm=student_llm, prompt=SQL_PROMPTS.get(valid_db.dialect, PROMPT))
        teacher_llm_chain = LLMChain(llm=teacher_llm, prompt=SQLITE_PROMPT_RETRY)
        return cls(student_llm_chain=student_llm_chain,
                   teacher_llm_chain=teacher_llm_chain,
                   validate_database=valid_db,
                   execute_database=remote_db,
                   **kwargs)
