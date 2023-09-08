## Release 1.3.0
### Major Features and Improvements
* FTL-LLM（Fedrated Learning + Transfer Learning + LLM）
  * Standard Offsite-Tuning and Extended Offsite-Tuning（Federated Offsite-Tuning+）now supported
  * Framework available for Emulator and Adapter development
  * New Offsite-Tuning Trainer introduced
  * Includes built-in models such as GPT-2 family, Llama7b, and Bloom family
* FedIPR
  * Introduced WatermarkDataset as the foundational dataset class for backdoor-based watermarks
  * Added SignConv and SignLayerNorm blocks for feature-based watermark models
  * New FedIPR Trainer available
  * Built-in models with feature-based watermarks include Alexnet, Resnet18, DistilBert, and GPT2
* More models support parameter-efficient fine-tuning: ChatGLM2-6B and Bloom-7B1


## Release 1.2.0
### Major Features and Improvements
* Support Federated Training of LLaMA-7B with parameter-efficient fine-tuning.


## Release 1.1.0
### Major Features and Improvements
* Support Federated Training of ChatGLM-6B with parameter-efficient fine-tuning adapters: like Lora and P-Tuning V2 etc.
* Integration of `peft`, which support many parameter-efficient adapters.
