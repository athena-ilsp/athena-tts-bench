"""
Greek TTS Text Preprocessing - LLM-Based Normalizer
Part of GreekTTS-Bench
Uses an LLM (OpenAI GPT-4, Anthropic Claude, or local model) for
context-aware text normalization.
"""

import json
import os
import re
from typing import List, Optional, Dict, Any, Tuple


# ── System Prompt (Greek) ─────────────────────────────────────────────────

SYSTEM_PROMPT = '''Είσαι ένα σύστημα προεπεξεργασίας κειμένου για Text-to-Speech (TTS) στα Ελληνικά. Στόχος σου είναι να μετατρέπεις γραπτό κείμενο σε μορφή έτοιμη για φυσική ανάγνωση από συνθέτη ομιλίας.

Ακολούθησε τους παρακάτω κανόνες με αυστηρή σειρά προτεραιότητας:

1. ΑΡΙΘΜΟΙ: Ανάπτυξε όλους τους αριθμούς σε λέξεις με το ΣΩΣΤΟ γραμματικό γένος σύμφωνα με το ουσιαστικό που συνοδεύουν.
   - "3 άνθρωποι" → "τρεις άνθρωποι" (αρσενικό)
   - "3 βιβλία" → "τρία βιβλία" (ουδέτερο)
   - "3 γυναίκες" → "τρεις γυναίκες" (θηλυκό)
   - "10.482.487 κάτοικοι" → "δέκα εκατομμύρια τετρακόσιες ογδόντα δύο χιλιάδες τετρακόσιοι ογδόντα επτά κάτοικοι"

2. ΤΑΚΤΙΚΟΙ ΑΡΙΘΜΟΙ: Ανάπτυξε τακτικούς αριθμούς (1ος, 2η, 3ο, 95η, κλπ):
   - "95η θέση" → "ενενηκοστή πέμπτη θέση"
   - "1ος αιώνας" → "πρώτος αιώνας"
   - "2ο βραβείο" → "δεύτερο βραβείο"

3. ΗΜΕΡΟΜΗΝΙΕΣ: Ανάπτυξε ημερομηνίες σε πλήρη φυσική μορφή:
   - "25η Μαρτίου 1821" → "είκοσι πέντε Μαρτίου χίλια οκτακόσια είκοσι ένα"
   - "25/03/1821" → "είκοσι πέντε Μαρτίου χίλια οκτακόσια είκοσι ένα"
   - "3 Φεβρουαρίου 1830" → "τρεις Φεβρουαρίου χίλια οκτακόσια τριάντα"

4. ΩΡΕΣ: Ανάπτυξε ώρες σε φυσική μορφή:
   - "09:00" → "εννέα ακριβώς" ή "εννέα το πρωί" (ανάλογα με τα συμφραζόμενα)
   - "14:30" → "δύο και μισή" ή "δεκατέσσερις και τριάντα" (διατήρησε το στυλ του κειμένου)
   - "17:30" → "πέντε και μισή"

5. ΠΟΣΟΣΤΑ: Ανάπτυξε ποσοστά:
   - "12,5%" → "δώδεκα και πέντε τοις εκατό"
   - "3,2%" → "τρία και δύο τοις εκατό"

6. ΝΟΜΙΣΜΑΤΑ: Ανάπτυξε χρηματικά ποσά:
   - "467,590 δισ. €" → "τετρακόσια εξήντα επτά δισεκατομμύρια πεντακόσια ενενήντα εκατομμύρια ευρώ"
   - "50 €" → "πενήντα ευρώ"

7. ΑΡΚΤΙΚΟΛΕΞΑ: Ανάπτυξε ΠΑΝΤΑ τα αρκτικόλεξα στην πλήρη ονομασία τους, στη ΣΩΣΤΗ πτώση, γένος και αριθμό ανάλογα με τη θέση τους στην πρόταση. ΠΟΤΕ μην αφήνεις το αρκτικόλεξο μέσα σε παρένθεση μετά την ανάπτυξη — το TTS θα διάβαζε ξανά τα γράμματα ένα προς ένα.
   - "του ΑΕΠ" → "του Ακαθάριστου Εγχώριου Προϊόντος"
   - "η ΕΚΤ" → "η Ευρωπαϊκή Κεντρική Τράπεζα"
   - "το ΔΝΤ" → "το Διεθνές Νομισματικό Ταμείο"  (ΟΧΙ: "το Διεθνές Νομισματικό Ταμείο (ΔΝΤ)")
   - "ο ΟΤΕ" → "ο Οργανισμός Τηλεπικοινωνιών Ελλάδος"
   - "η ΔΕΗ" → "η Δημόσια Επιχείρηση Ηλεκτρισμού"
   - "ο ΟΗΕ" → "ο Οργανισμός Ηνωμένων Εθνών"
   - "ΤΝ" → "τεχνητή νοημοσύνη"
   - "ΕΛΣΤΑΤ" → "Ελληνική Στατιστική Αρχή"
   ΕΞΑΙΡΕΣΗ — αρκτικόλεξα που προφέρονται ως κανονική λέξη: άφησέ τα ως έχουν, ΧΩΡΙΣ ανάπτυξη και ΧΩΡΙΣ παρένθεση:
   - "ΝΑΤΟ" → "ΝΑΤΟ"
   - "ΟΥΝΕΣΚΟ" → "ΟΥΝΕΣΚΟ"
   - "ΚΤΕΛ" → "ΚΤΕΛ"

8. ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ: Ανάπτυξε όλες τις συντομογραφίες:
   - "κ.λπ." → "και λοιπά"
   - "π.χ." → "παραδείγματος χάριν"
   - "δηλ." → "δηλαδή"
   - "κ.ά." → "και άλλα"
   - "π.Χ." → "προ Χριστού"
   - "μ.Χ." → "μετά Χριστόν"
   - "π.μ." → "προ μεσημβρίας"
   - "μ.μ." → "μετά μεσημβρίαν"
   - "χλμ." → "χιλιόμετρα"
   - "εκ." → "εκατομμύρια"
   - "δισ." → "δισεκατομμύρια"
   - "κ." → "κύριος" (όταν αναφέρεται σε πρόσωπο)
   - "κα." → "κυρία"
   - "σελ." → "σελίδα"

9. ΑΓΓΛΙΚΕΣ ΛΕΞΕΙΣ ΣΕ ΕΛΛΗΝΙΚΟ ΚΕΙΜΕΝΟ: Για αγγλικές λέξεις/φράσεις μέσα σε ελληνικό κείμενο, αν το κείμενο προορίζεται για TTS που υποστηρίζει αγγλικά, άφησέ τες ως έχουν. Αν όχι, πρόσθεσε ελληνική φωνητική απόδοση σε παρένθεση:
   - "Το software εγκαταστάθηκε" → "Το software (σόφτγουερ) εγκαταστάθηκε"

10. ΣΗΜΕΙΑ ΣΤΙΞΗΣ: Διατήρησε όλα τα σημεία στίξης ακριβώς όπως είναι (τελείες, κόμματα, ερωτηματικά, θαυμαστικά, εισαγωγικά, παρενθέσεις). Χρησιμοποιούνται από το TTS για τον καθορισμό της προσωδίας.

11. ΥΠΟΛΟΙΠΟ ΚΕΙΜΕΝΟ: ΜΗΝ αλλάξεις τίποτα άλλο στο κείμενο. ΜΗΝ προσθέσεις επεξηγήσεις, σχόλια ή δικές σου προτάσεις. Απλά επέστρεψε το κανονικοποιημένο κείμενο.

Παράδειγμα εισόδου:
"Ο πληθυσμός της Ελλάδας ανέρχεται σε 10.482.487 κατοίκους, σύμφωνα με την ΕΛΣΤΑΤ (απογραφή 2021). Το ΑΕΠ αυξήθηκε κατά 3,2% το 2023."

Παράδειγμα εξόδου:
"Ο πληθυσμός της Ελλάδας ανέρχεται σε δέκα εκατομμύρια τετρακόσιες ογδόντα δύο χιλιάδες τετρακόσιους ογδόντα επτά κατοίκους, σύμφωνα με την Ελληνική Στατιστική Αρχή (απογραφή δύο χιλιάδες είκοσι ένα). Το Ακαθάριστο Εγχώριο Προϊόν αυξήθηκε κατά τρία και δύο τοις εκατό το δύο χιλιάδες είκοσι τρία."'''


# ── Output cleaning and validation ────────────────────────────────────────
# Smaller self-hosted models (e.g. Krikri-8B) sometimes wrap the answer in
# quotes/markdown, prepend a label, or refuse. Whatever survives here goes
# straight to the TTS systems, so clean it and fall back to the original text
# when the output looks unusable. Warnings are recorded for spot-checking.

_OUTPUT_LABEL_RE = re.compile(
    r"^\s*(?:Παράδειγμα\s+εξόδου|Έξοδος|Κανονικοποιημένο\s+κείμενο|Output|"
    r"Normalized\s+text|Απάντηση)\s*[:：]\s*",
    re.IGNORECASE,
)

_REFUSAL_MARKERS = (
    "δεν μπορώ",
    "δε μπορώ",
    "λυπάμαι",
    "ως μοντέλο",
    "ως γλωσσικό μοντέλο",
    "i cannot",
    "i can't",
    "i'm sorry",
    "as an ai",
    "as a language model",
)

# Expansion can only grow the text; a much shorter or wildly longer output
# means the model did something other than normalize.
_MIN_LENGTH_RATIO = 0.5
_MAX_LENGTH_RATIO = 5.0


def clean_llm_output(text: str) -> str:
    """Strip wrappers the model may add around the normalized text."""
    cleaned = text.strip()

    # Markdown code fences.
    fence = re.match(r"^```[\w-]*\n(.*?)\n?```\s*$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Leading labels like "Έξοδος:".
    cleaned = _OUTPUT_LABEL_RE.sub("", cleaned)

    # Surrounding quotes added by the model (only when wrapping everything).
    for opening, closing in (('"', '"'), ("«", "»"), ("“", "”"), ("'", "'")):
        if (
            len(cleaned) >= 2
            and cleaned.startswith(opening)
            and cleaned.endswith(closing)
            and cleaned.count(opening) == (1 if opening != closing else 2)
        ):
            cleaned = cleaned[1:-1].strip()

    return cleaned


def validate_llm_output(original: str, cleaned: str) -> List[str]:
    """Return warnings describing why the output looks unusable."""
    warnings: List[str] = []
    if not cleaned:
        warnings.append("empty output")
        return warnings

    lowered = cleaned.lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lowered and marker not in original.lower():
            warnings.append(f"possible refusal/meta text: {marker!r}")
            break

    ratio = len(cleaned) / max(len(original), 1)
    if ratio < _MIN_LENGTH_RATIO:
        warnings.append(f"output too short (ratio {ratio:.2f})")
    elif ratio > _MAX_LENGTH_RATIO:
        warnings.append(f"output too long (ratio {ratio:.2f})")

    if cleaned.count("\n") > original.count("\n") + 2:
        warnings.append("output has unexpected extra lines")

    return warnings


# A parenthetical that is only uppercase Greek letters (optionally separated by
# dots/spaces), e.g. "(ΔΝΤ)" or "(Α.Ε.)". Models like to expand an acronym and
# then re-state it in parentheses ("Διεθνές Νομισματικό Ταμείο (ΔΝΤ)"); for TTS
# that is the worst case, because the synth reads the letters again. Legitimate
# spoken parentheticals contain lowercase words, so they never match this.
_PAREN_ACRONYM_RE = re.compile(r"\s*\(\s*[Α-ΩΪΫ](?:[.\s]*[Α-ΩΪΫ])+\s*\.?\s*\)")


def strip_parenthetical_acronyms(text: str) -> str:
    """Remove "(ΑΚΡ)"-style uppercase-Greek acronyms left in parentheses.

    Deterministic guardrail applied after the LLM: even when the prompt fails to
    stop the model from re-stating an expanded acronym in parentheses, this keeps
    those letters out of the text the TTS reads.
    """
    cleaned = _PAREN_ACRONYM_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Repair a space stranded before sentence punctuation by a removed group.
    cleaned = re.sub(r"\s+([,.;:!»)])", r"\1", cleaned)
    return cleaned.strip()


# ── LLM Preprocessor Class ────────────────────────────────────────────────

class LLMPreprocessor:
    """LLM-based Greek text normalizer for TTS.

    Supports multiple LLM backends:
    - OpenAI (GPT-4o, GPT-4)
    - Anthropic (Claude)
    - Any OpenAI-compatible API (local vLLM, Ollama, etc.)
    - transformers: a self-hosted causal LM loaded in-process, no network/API.
      Suited to offline HPC nodes; the default Greek target is
      Llama-Krikri-8B-Instruct (ilsp/Llama-Krikri-8B-Instruct).
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o",
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 device_map: str = "auto", torch_dtype: str = "auto"):
        """
        Args:
            provider: "openai", "anthropic", "local", or "transformers"
            model: Model name or path (e.g., "gpt-4o",
                "claude-3-5-sonnet-20241022", or a Hugging Face id/path like
                "ilsp/Llama-Krikri-8B-Instruct" for provider="transformers")
            api_key: API key (defaults to env var based on provider)
            base_url: Custom API base URL (for local OpenAI-compatible servers)
            temperature: LLM temperature (0.0 for deterministic output)
            max_tokens: Max tokens / max new tokens in the response
            device_map: Device placement for provider="transformers"
            torch_dtype: Torch dtype for provider="transformers"
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self._client = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        if provider == "transformers":
            self._init_transformers()
        else:
            self._client = self._init_client()

    def _init_transformers(self):
        """Load a self-hosted causal LM in-process (no network/API).

        Suited to offline HPC nodes. The default Greek target is the
        Greek-specialized Llama-Krikri-8B-Instruct.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "provider='transformers' needs: transformers, torch, accelerate. "
                "Install them or set up the environment on the node."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, torch_dtype=self.torch_dtype, device_map=self.device_map
        )
        self._model.eval()

    def _init_client(self):
        """Initialize the appropriate API client."""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                return OpenAI(
                    api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                    base_url=self.base_url,
                )
            elif self.provider == "anthropic":
                import anthropic
                return anthropic.Anthropic(
                    api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif self.provider == "local":
                from openai import OpenAI
                return OpenAI(
                    api_key=self.api_key or "not-needed",
                    base_url=self.base_url or "http://localhost:11434/v1",
                )
        except ImportError:
            print(f"Warning: Could not import client for {self.provider}. "
                  f"Install with: pip install openai anthropic")
            return None

    def _generate_raw(self, text: str, prompt: str) -> str:
        """Run one normalization request and return the raw model output."""
        if self.provider == "transformers":
            return self._normalize_transformers(text, prompt)

        if self._client is None:
            raise RuntimeError(f"No API client initialized for {self.provider}")

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt,
                messages=[{"role": "user", "content": text}]
            )
            return response.content[0].text

        else:
            # OpenAI and OpenAI-compatible APIs
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content

    def normalize_with_report(
        self, text: str, system_prompt: str = None
    ) -> Tuple[str, List[str]]:
        """Normalize one text and report output-quality warnings.

        The raw model output is cleaned (fences, labels, wrapping quotes).
        If the cleaned output still looks unusable (empty, refusal, length way
        off), the original text is returned instead so the TTS never reads
        model chatter; the warnings say why.
        """
        prompt = system_prompt or SYSTEM_PROMPT
        raw = self._generate_raw(text, prompt) or ""
        cleaned = clean_llm_output(raw)
        cleaned = strip_parenthetical_acronyms(cleaned)
        warnings = validate_llm_output(text, cleaned)
        if warnings:
            return text, warnings
        return cleaned, []

    def normalize(self, text: str, system_prompt: str = None) -> str:
        """Normalize a single text string for TTS.

        Args:
            text: Input Greek text to normalize
            system_prompt: Custom system prompt (uses default if None)

        Returns:
            Normalized text ready for TTS (original text if the model output
            was unusable; see normalize_with_report for details)
        """
        normalized, _ = self.normalize_with_report(text, system_prompt)
        return normalized

    def _normalize_transformers(self, text: str, prompt: str) -> str:
        """Normalize one text with the in-process transformers model."""
        torch = self._torch
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)

        gen_kwargs = {"max_new_tokens": self.max_tokens}
        if self.temperature and self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0, prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def normalize_dataset(self, input_json_path: str, output_json_path: str,
                          batch_size: int = 10) -> Dict[str, Any]:
        """Normalize all sentences in a dataset JSON file.

        Args:
            input_json_path: Path to input dataset JSON
            output_json_path: Path to save normalized dataset
            batch_size: Number of sentences per API call (sent individually for better quality)

        Returns:
            Updated dataset dict
        """
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data['sentences'])
        warning_count = 0
        for i, sentence in enumerate(data['sentences']):
            print(f"  [{i+1}/{total}] Normalizing: {sentence['id']}")
            sentence['original_text'] = sentence['text']
            normalized, warnings = self.normalize_with_report(sentence['text'])
            sentence['normalized_text'] = normalized
            if warnings:
                sentence['normalization_warnings'] = warnings
                warning_count += 1
                print(f"    WARNING {sentence['id']}: {'; '.join(warnings)}")

        data['preprocessing'] = {
            'method': 'llm',
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'sentences_with_warnings': warning_count,
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Demo with example sentences (no API call needed for demo)
    demo_sentences = [
        "Ο πληθυσμός της Ελλάδας ανέρχεται σε 10.482.487 κατοίκους, σύμφωνα με την ΕΛΣΤΑΤ (απογραφή 2021).",
        "Η ανεξαρτησία κηρύχθηκε την 25η Μαρτίου 1821 και αναγνωρίστηκε στις 3 Φεβρουαρίου 1830.",
        "Το ΑΕΠ αυξήθηκε κατά 3,2% το 2023, σύμφωνα με το ΔΝΤ.",
        "Η Ελλάδα είναι μέλος του ΝΑΤΟ από το 1952 και ιδρυτικό μέλος του ΟΗΕ, του ΟΟΣΑ και του ΠΟΕ.",
        "Το software εγκαταστάθηκε στον server και το backup ολοκληρώθηκε επιτυχώς.",
        "Βλ. επίσης Μπίλλης (2025), σελ. 45-67, κ.ά. - η συνάντηση είναι στις 14:30.",
    ]

    print("=" * 70)
    print("LLMPreprocessor - Demo (no API calls)")
    print("=" * 70)
    print("\nThis module normalizes Greek text for TTS using an LLM.")
    print("To use it with a real API, run:\n")
    print("  python llm_preprocessor.py <input.json> <output.json>")
    print("\nExample sentences that would be normalized:\n")

    for i, s in enumerate(demo_sentences, 1):
        print(f"  {i}. {s}")

    print("\n" + "=" * 70)
    print("\nSupported providers:")
    print("  - OpenAI:     export OPENAI_API_KEY=sk-...")
    print("  - Anthropic:  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  - Local:      Ollama/vLLM on http://localhost:11434/v1")

    # If file arguments provided, run the normalizer
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_llm_normalized.json')
        provider = sys.argv[3] if len(sys.argv) > 3 else "openai"
        model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o"

        print(f"\nRunning LLM normalizer: provider={provider}, model={model}")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        processor = LLMPreprocessor(provider=provider, model=model)
        processor.normalize_dataset(input_file, output_file)
        print("Done!")
