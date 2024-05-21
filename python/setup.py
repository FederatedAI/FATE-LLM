# -*- coding: utf-8 -*-

from setuptools import setup

packages = ["fate_llm", "fate_llm.evaluate.scripts"]
entry_points = {"console_scripts": ["fate_llm = fate_llm.evaluate.scripts.fate_llm_cli:fate_llm_cli"]}

setup(
    name='fate_llm',
    version='2.2.0',
    packages=packages,
    entry_points=entry_points
)
