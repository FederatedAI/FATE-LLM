# FATE-LLM Single-Node Deployment Guide

## 1. Introduction

**Server Configuration:**

- **Quantity:** 1
- **Configuration:** 8 cores / 16GB memory / 500GB hard disk / GPU Machine
- **Operating System:** CentOS Linux release 7
- **User:** User: app owner:apps

The single-node version provides 3 deployment methods, which can be selected based on your needs:
- Install FATE-LLM from PyPI With FATE
- Install FATE-LLM from PyPI with FATE, FATE-Flow, FATE-Client

## 2. Install FATE-LLM from PyPI With FATE
In this way, user can run tasks with Launcher, a convenient way for fast experimental using.

### 2.1 Installing Python Environment
- Prepare and install [conda](https://docs.conda.io/projects/miniconda/en/latest/) environment.
- Create a virtual environment:

```shell
# FATE-LLM requires Python >= 3.10
conda create -n fate_env python=3.10
conda activate fate_env
```

### 2.2 Installing FATE-LLM
This section introduces how to install FATE-LLM from pypi with FATE, execute the following command to install FATE-LLM. 

```shell
pip install fate_llm[fate]==2.2.0
```

### 2.3 Usage
After installing successfully, please refer to [tutorials](../README.md#quick-start) to run tasks, tasks describe in the tutorials running will Launcher are all supported.


## 3. Install FATE-LLM from PyPI with FATE, FATE-Flow, FATE-Client
In this way, user can run tasks with Pipeline or Launcher. 

### 3.1 Installing Python Environment
Please refer to section-2.1

### 3.2 Installing FATE-LLM with FATE, FATE-Flow, FATE-Client

```shell
pip install fate_client[fate,fate_flow,fate_client]==2.2.0
```

### 3.3 Service Initialization

```shell
mkdir fate_workspace
fate_flow init --ip 127.0.0.1 --port 9380 --home $(pwd)/fate_workspace
pipeline init --ip 127.0.0.1 --port 9380
```
- `ip`: The IP address where the service runs.
- `port`: The HTTP port the service runs on.
- `home`: The data storage directory, including data, models, logs, job configurations, and SQLite databases.

### 3.4 Start Fate-Flow Service

```shell
fate_flow start
fate_flow status # make sure fate_flow service is started
```

FATE-Flow also provides other instructions like stop and restart, use only if users want to stop/restart fate_flow services.
```shell
# Warning: normal installing process does not need to execute stop/restart instructions.
fate_flow stop
fate_flow restart
```

### 3.5 Usage
Please refer to [tutorials](../README.md#quick-start) for more usage guides, tasks describe in the tutorials running will Pipeline or Launcher are all supported.
