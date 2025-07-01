"""
Resume-Builder – end-to-end pipeline
-----------------------------------
Python ≥ 3.9
pip install pdf2image pytesseract pillow langgraph ollama-python pyyaml
"""

# ------------------------------------------------------------------------
# Guarantee Annotated is always available to get_type_hints(), even if an
# old .pyc is accidentally loaded or the import fails.
# ------------------------------------------------------------------------
try:
    from typing import Annotated  # Python 3.10 +
except ImportError:               # extremely old envs
    from typing_extensions import Annotated  # back-port

globals()["Annotated"] = Annotated            # ← make sure it's in globals()


import argparse
import json
import os
import pprint
from typing import Any, Dict, Optional, Annotated, TypedDict

import yaml                                         # pip install pyyaml
import importlib.util
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

# ────────────────────────────────────────────────────────────────────────────
# 1. Load configuration (config.yml)
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
CFG_PATH = os.path.join(SCRIPT_DIR, "config.yml")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

# ────────────────────────────────────────────────────────────────────────────
# 2. Dynamic import of ocr_utils.py   (must expose pdf_to_text)
# ────────────────────────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "ocr_utils", os.path.join(SCRIPT_DIR, "ocr_utils.py")
)
ocr_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ocr_utils)
extract_text_from_pdf = ocr_utils.pdf_to_text  # alias

# ────────────────────────────────────────────────────────────────────────────
# 3. Prompt template (from YAML)
# ────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: Dict[str, str] = CONFIG["prompt_templates"]


def build_prompt(resume_text: str, jd_text: str = "", resume_style: str = "modern") -> str:
    """Fill the template; use (none) placeholders when text is missing."""
    template = PROMPT_TEMPLATES.get(resume_style, PROMPT_TEMPLATES[resume_style])
    return template.format(
        resume_text=resume_text or "(none)",
        job_description_text=jd_text or "(none)",
    )


# ────────────────────────────────────────────────────────────────────────────
# 4. LLM via Ollama
# ────────────────────────────────────────────────────────────────────────────
import ollama
import openai

DEFAULT_MODEL_TYPE = "Free (Ollama gemma2:9b)"


def call_openai(prompt: str, model: str) -> Dict[str, Any]:
    """Run OpenAI model and parse strict-JSON response."""
    client = openai.OpenAI(api_key=CONFIG["openai"]["api_key"])
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def call_llm(prompt: str, model_type: str = DEFAULT_MODEL_TYPE) -> Dict[str, Any]:
    """Run local Ollama model and parse strict-JSON response."""
    model_info = CONFIG["model_types"][model_type]
    provider = model_info["provider"]
    model = model_info["model"]

    if provider == "ollama":
        completion = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            format="json",  # force model to return JSON string
        )["message"]["content"]
        try:
            return json.loads(completion)
        except json.JSONDecodeError as err:
            raise ValueError(
                "LLM did not return valid JSON. Raw output was:\n" + completion
            ) from err
    elif provider == "openai":
        return call_openai(prompt, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")



# ────────────────────────────────────────────────────────────────────────────
# 5. LangGraph orchestration
# ────────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END


def _override(left, right):  # reducer: last value wins
    return right



class State(TypedDict, total=False):
    # input params
    resume_path: str
    job_description_text: str|None  # ← JD is now plain text, not path
    poppler_path: str
    dpi: int
    lang: str
    model_type: str
    resume_style: str

    # working fields
    resume_text: Annotated[str, _override]
    prompt: Annotated[str, _override]
    pdf_warning: Annotated[str, _override]

    # final output
    llm_output: Dict[str, Any]


def build_graph() -> StateGraph:
    g = StateGraph(State)

    # ── Node 1: OCR Résumé ────────────────────────────────────────────────
    def ocr_resume(state: State) -> State:
        try:
            text = extract_text_from_pdf(
                state["resume_path"],
                poppler_path=state.get("poppler_path"),
                dpi=state["dpi"],
                lang=state["lang"],
                keep_linebreaks=True,
            )
            return {"resume_text": text}
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            return {
                "resume_text": "",
                "pdf_warning": f"OCR skipped for résumé: {e}",
            }

    # ── Node 2: Build prompt ─────────────────────────────────────────────
    def make_prompt(state: State) -> State:
        return {
            "prompt": build_prompt(
                state["resume_text"],
                state.get("job_description_text", ""),
                state.get("resume_style", "modern"),
            )
        }

    # ── Node 3: LLM ───────────────────────────────────────────────────────
    def llm_node(state: State) -> State:
        output = call_llm(state["prompt"], model_type=state["model_type"])
        if "pdf_warning" in state and state["pdf_warning"]:
            output["pdf_warning"] = state["pdf_warning"]
        return {"llm_output": output}

    # Wire graph
    g.add_node("ocr_resume", ocr_resume)
    g.add_node("make_prompt", make_prompt)
    g.add_node("llm_node", llm_node)

    g.set_entry_point("ocr_resume")
    g.add_edge("ocr_resume", "make_prompt")
    g.add_edge("make_prompt", "llm_node")
    g.add_edge("llm_node", END)

    return g


# ────────────────────────────────────────────────────────────────────────────
# 6. Runner helpers + CLI
# ────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    resume_path: str,
    job_description_text: Optional[str],
    poppler_path: Optional[str],
    model_type: str,
    dpi: int,
    lang: str,
    resume_style: str,
    *,
    return_state: bool = False,
) -> Any:
    compiled = build_graph().compile()
    state = compiled.invoke(
        {
            "resume_path": resume_path,
            "job_description_text": job_description_text,
            "poppler_path": poppler_path,
            "model_type": model_type,
            "dpi": dpi,
            "lang": lang,
            "resume_style": resume_style,
        }
    )
    return state if return_state else state["llm_output"]


def main() -> None:
    parser = argparse.ArgumentParser(description="ATS Résumé Builder")
    parser.add_argument("resume", help="Path to résumé PDF")
    parser.add_argument(
        "-j",
        "--job-text",
        help="Job description (plain text, optional)",
    )
    parser.add_argument(
        "-p",
        "--poppler",
        default=CONFIG.get("poppler_path"),
        help="Poppler bin path (overrides YAML)",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        choices=CONFIG["model_types"].keys(),
        help="Model type to use",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="modern",
        choices=CONFIG.get("resume_styles", ["modern"]),
        help="Résumé style to generate",
    )
    parser.add_argument("--dpi", type=int, default=CONFIG.get("dpi", 300))
    parser.add_argument("--lang", default=CONFIG.get("lang", "eng"))
    args = parser.parse_args()

    result = run_pipeline(
        resume_path=args.resume,
        job_description_text=args.job_text,
        poppler_path=args.poppler,
        model_type=args.model_type,
        dpi=args.dpi,
        lang=args.lang,
        resume_style=args.style,
    )
    pprint.pprint(result, width=120)


if __name__ == "__main__":
    main()
