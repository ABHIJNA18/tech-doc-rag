# ---------------------------------------------------------
# prompts.py
#
# Loads versioned prompts from configs/prompts.yaml and turns them
# into LangChain ChatPromptTemplates.
#
# Usage:
#   prompt_template, version = load_prompt_template("answer_with_citations")
#   messages = prompt_template.invoke({"sources": ..., "question": ...})
# ---------------------------------------------------------

from __future__ import annotations

from functools import lru_cache

import yaml
from langchain_core.prompts import ChatPromptTemplate

from rag import config


@lru_cache(maxsize=None)
def load_prompt_file() -> dict:
    with config.PROMPTS_FILE.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_prompt_template(prompt_name: str, version: str | None = None) -> tuple[ChatPromptTemplate, str]:
    """
    The chat prompt (system + user message) for a prompt name and
    version (default: config.PROMPT_VERSION). Returns the template and
    the version used, so callers can record it.
    """

    version = version or config.PROMPT_VERSION
    versions = load_prompt_file().get(prompt_name, {})

    if version not in versions:
        raise ValueError(f"Prompt {prompt_name!r} has no version {version!r}. Available: {', '.join(versions)}")

    prompt_config = versions[version]

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt_config["system"]), ("human", prompt_config["user"])]
    )

    return prompt_template, version
