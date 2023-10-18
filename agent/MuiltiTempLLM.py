# from __future__ import annotations
import openai
import json
import os
from langchain.pydantic_v1 import root_validator
from langchain.llms import OpenAI
from langchain.llms.openai import BaseOpenAI, OpenAIChat
from typing import Dict, Any, List, Optional, Sequence, Union
import warnings


class MultiTempLLM:
    """
    Add set function to modify temperature to adapt to different usage

    """

    @classmethod
    def load_api_keys(cls, engine_type="gpt-4-canada") -> List:
        """
        :param engine_type: choose from "gpt-3.5", "gpt-4-8k","gpt-4-canada"
        :return: List of different deploys and models
        """
        project_path = os.path.dirname(os.getcwd())
        config_file_path = os.path.join(project_path, "keys/api_instances.json")
        with open(config_file_path, "r") as f:
            instances = json.load(f)
        f.close()

        assert engine_type in instances.keys(), "Engine type do not support!"

        resource = instances[engine_type]

        openai.api_type = resource["api_type"]
        openai.api_base = resource["api_base"]
        os.environ["OPENAI_API_VERSION"] = resource["api_version"]
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        os.environ["OPENAI_API_KEY"] = resource["api_key"]
        openai.api_key = os.getenv("OPENAI_API_KEY")

        return resource["deploy_list"]

    @classmethod
    def check_keys(cls, values: Dict) -> Dict:
        if "engine" not in values.keys():
            values["engine"] = values.pop("deploy_name")
        if "expire_date" in values.keys():
            values.pop("expire_date")
        if "max_speed" in values.keys():
            values.pop("max_speed")
        return values

    @classmethod
    def check_model_type(cls, model_name) -> bool:
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            return True
        else:
            return False

    def __new__(cls, **values) -> Union[BaseOpenAI, OpenAIChat]:
        values = cls.check_keys(values)
        if cls.check_model_type(values.get("model_name")):
            return OpenAIChat(**values)
        else:
            return BaseOpenAI(**values)

    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if self.__class__.__name__ == "OpenAIChat":
            self.invoke(text)
        else:
            self.predict(text)

    def __str__(self):
        return "Instance class name: %s, Model kwargs: %s" % (
            self.__class__.__name__,
            self.model_kwargs)


if __name__ == "__main__":
    deploys = MultiTempLLM.load_api_keys(engine_type="gpt-4-8k")
    # print(deploys)
    llm = MultiTempLLM(**deploys[1], temperature=0.2)
    print(llm)
    res = llm.predict("hello! Tell me a joke for test.")
    print(res)
