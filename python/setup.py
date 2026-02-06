# -*- coding: utf-8 -*-
# 
#  Copyright 2024 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from setuptools import find_packages, setup

# Define the packages and modules
packages = find_packages(".")
package_data = {"": ["*"]}

# Define dependencies
install_requires = [
    "accelerate==0.27.2",
    "deepspeed==0.13.3", 
    "peft==0.8.2",
    "sentencepiece==0.2.0",
    "lm_eval==0.4.2",
    "rouge-score==0.1.2",
    "datasets==2.18.0",
    "editdistance",
    "torch==2.3.1",
    "transformers==4.37.2",
    "opacus==1.4.1",
    "fastchat",
    "Jinja2",
    "sentence-transformers",
    "openai"
]

# Define the entry points for command-line tools
entry_points = {
    "console_scripts": [
        "fate_llm = fate_llm.evaluate.scripts.fate_llm_cli:fate_llm_cli"
    ]
}

extras_require = {
    "fate": ["pyfate==2.2.0"],
    "fate_flow": ["fate_flow==2.2.0"],
    "fate_client": ["fate_client==2.2.0"]
}

# Configure and call the setup function
setup_kwargs = {
    "name": "fate_llm",
    "version": "2.2.0",
    "description": "Federated Learning for Large Language Models",
    "long_description": "Federated Learning for Large Language Models (FATE-LLM) provides a framework to train and evaluate large language models in a federated manner.",
    "long_description_content_type": "text/markdown",
    "author": "FederatedAI",
    "author_email": "contact@FedAI.org",
    "url": "https://fate.fedai.org/",
    "packages": packages,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "extras_require": extras_require,
    "python_requires": ">=3.8",
    "include_package_data": True
}

setup(**setup_kwargs)
