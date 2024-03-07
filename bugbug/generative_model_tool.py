# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from typing import Any
from langchain_openai import ChatOpenAI


class GenerativeModelTool(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.2)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        ...
