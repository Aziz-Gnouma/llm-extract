"""llmx: structured entity extraction with a QLoRA-fine-tuned small LLM.

Layering (imports only point downwards):

    pipeline / cli            orchestration, file I/O of artefacts
    training, inference.generate, modeling   GPU code (torch / transformers / peft)
    data, inference.parsing, evaluation      pure Python, unit-tested without a GPU
    config                    typed configuration
"""

__version__ = "0.1.0"
