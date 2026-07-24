"""System-prompt archetypes.

msqa/general/choice are verbatim from HORNet (reference/hornet/util.py,
qwen_answer_question); yesno follows TimeBlind's protocol (no system prompt). Bishoy's
guidance: pick/adapt the archetype that matches the benchmark's QA type, do not reuse a
mismatched one.

  msqa    -> one-word answers, scored with F1-Lev       (MSVD / MSRVTT style)
  general -> short-sentence answers, scored with F1-Lev (NExT-QA open-ended style)
  choice  -> digit-index answers, scored with accuracy  (NExT-QA MCQ / VideoMME style)
  yesno   -> no system prompt; format instruction appended to the question (TimeBlind)

For `choice`, the user prompt must append the options exactly like HORNet does:
    question + "\\n Your choices are: " + ", ".join(options)
(the runner does this automatically when prompt.task == "choice").
"""

ARCHETYPES = {
    "msqa": "Based on the video, please answer the question in only one word and end with a period ('.'). ",
    "general": "Based on the video, please give answer in a short sentence. Your first sentence should be the concise answer.",
    "choice": "Based on the video, please answer the question by selecting the correct index, output only one digit. ",
    # TimeBlind official protocol (reference/timeblind/utils.py add_question_suffix):
    # NO system prompt — the format instruction is a suffix appended to the question.
    "yesno": "",
    # Contrastive two-video first/second task: no system prompt; the full
    # comparative question (with its "answer only first/second" instruction) is the user prompt.
    "contrastive": "",
}


def build_user_prompt(sample, task: str) -> str:
    """Compose the final user prompt exactly like the benchmark's own eval scripts."""
    prompt = sample.question or ""
    if task == "choice" and sample.options:
        prompt = prompt + "\n Your choices are: " + ", ".join(str(o) for o in sample.options)
    elif task == "yesno":
        # official TimeBlind add_question_suffix — type-aware
        if (sample.category or "yes_no") == "multiple_choice":
            prompt = prompt + " Please output A or B."
        else:
            prompt = prompt + " Please output Yes or No."
    return prompt


def resolve_system_prompt(cfg) -> tuple[str, str]:
    """Returns (task, system_prompt). A custom file/inline prompt overrides the archetype."""
    task = (cfg.prompt.get("task") or "general").lower()
    if task not in ARCHETYPES and task != "custom":
        raise ValueError(f"prompt.task must be one of {sorted(ARCHETYPES)} or 'custom', got '{task}'")
    custom = cfg.system_prompt()  # from system_prompt_file / system_prompt keys, may be ""
    if custom:
        return task, custom
    if task == "custom":
        raise ValueError("prompt.task is 'custom' but no system_prompt(_file) is set")
    return task, ARCHETYPES[task]  # "" for yesno => adapters send no system message
