"""
Greek TTS Text Preprocessing - Regex-Based Normalizer
Part of GreekTTS-Bench
Handles: numbers, dates, times, percentages, currencies, acronyms, abbreviations
"""

import re
from typing import Dict, Tuple


# ── Greek Acronym Dictionary ──────────────────────────────────────────────
GREEK_ACRONYM_MAP: Dict[str, str] = {
    # International organizations & bodies
    "ΟΗΕ": "Οργανισμός Ηνωμένων Εθνών",
    "ΝΑΤΟ": "ΝΑΤΟ",  # pronounced as word /nato/
    "ΟΟΣΑ": "Οργανισμός Οικονομικής Συνεργασίας και Ανάπτυξης",
    "ΠΟΕ": "Παγκόσμιος Οργανισμός Εμπορίου",
    "ΟΑΣΕ": "Οργανισμός για την Ασφάλεια και τη Συνεργασία στην Ευρώπη",
    "ΔΝΤ": "Διεθνές Νομισματικό Ταμείο",
    "UNESCO": "ΟΥΝΕΣΚΟ",
    "UNICEF": "Γιούνισεφ",

    # European Union
    "ΕΕ": "Ευρωπαϊκή Ένωση",
    "ΕΚΤ": "Ευρωπαϊκή Κεντρική Τράπεζα",
    "ΕΣΠΑ": "Εταιρικό Σύμφωνο για το Πλαίσιο Ανάπτυξης",

    # Greek government & public sector
    "ΕΛΣΤΑΤ": "Ελληνική Στατιστική Αρχή",
    "ΔΕΗ": "Δημόσια Επιχείρηση Ηλεκτρισμού",
    "ΟΤΕ": "Οργανισμός Τηλεπικοινωνιών Ελλάδος",
    "ΕΥΔΑΠ": "Εταιρεία Υδρεύσεως και Αποχετεύσεως Πρωτευούσης",
    "ΕΟΠΥΥ": "Εθνικός Οργανισμός Παροχής Υπηρεσιών Υγείας",
    "ΑΔΑΕ": "Αρχή Διασφάλισης του Απορρήτου των Επικοινωνιών",
    "ΕΣΡ": "Εθνικό Συμβούλιο Ραδιοτηλεόρασης",
    "ΕΡΤ": "Ελληνική Ραδιοφωνία Τηλεόραση",
    "ΓΓΠΣ": "Γενική Γραμματεία Πληροφοριακών Συστημάτων",
    "ΙΚΑ": "Ίδρυμα Κοινωνικών Ασφαλίσεων",
    "ΕΦΚΑ": "Ενιαίος Φορέας Κοινωνικής Ασφάλισης",
    "ΑΜΚΑ": "Αριθμός Μητρώου Κοινωνικής Ασφάλισης",
    "ΥΠΕΘΑ": "Υπουργείο Εθνικής Άμυνας",
    "ΥΠΟΙΚ": "Υπουργείο Οικονομικών",
    "ΥΠΕΝ": "Υπουργείο Περιβάλλοντος και Ενέργειας",
    "ΥΠΑΙΘ": "Υπουργείο Παιδείας και Θρησκευμάτων",
    "ΥΠΕΞ": "Υπουργείο Εξωτερικών",

    # Political parties
    "ΝΔ": "Νέα Δημοκρατία",
    "ΠΑΣΟΚ": "ΠΑΣΟΚ",
    "ΣΥΡΙΖΑ": "ΣΥΡΙΖΑ",
    "ΚΚΕ": "Κομμουνιστικό Κόμμα Ελλάδας",
    "ΜέΡΑ25": "Μέτωπο Ευρωπαϊκής Ρεαλιστικής Ανυπακοής",

    # Economic/financial
    "ΑΕΠ": "Ακαθάριστο Εγχώριο Προϊόν",
    "ΦΠΑ": "Φόρος Προστιθέμενης Αξίας",
    "ΕΝΦΙΑ": "Ενιαίος Φόρος Ιδιοκτησίας Ακινήτων",
    "ΧΑΑ": "Χρηματιστήριο Αξιών Αθηνών",

    # Education
    "ΑΕΙ": "Ανώτατο Εκπαιδευτικό Ίδρυμα",
    "ΤΕΙ": "Τεχνολογικό Εκπαιδευτικό Ίδρυμα",
    "ΙΕΚ": "Ινστιτούτο Επαγγελματικής Κατάρτισης",

    # Technical
    "ΤΝ": "τεχνητή νοημοσύνη",
    "Η/Υ": "ηλεκτρονικός υπολογιστής",
    "GPS": "τζι-πι-ες",

    # Media
    "ΑΠΕ-ΜΠΕ": "Αθηναϊκό Πρακτορείο Ειδήσεων - Μακεδονικό Πρακτορείο Ειδήσεων",

    # Countries
    "ΗΠΑ": "Ηνωμένες Πολιτείες Αμερικής",
    "ΗΒ": "Ηνωμένο Βασίλειο",
    "πΓΔΜ": "πρώην Γιουγκοσλαβική Δημοκρατία της Μακεδονίας",
}


# ── Greek Abbreviation Dictionary ─────────────────────────────────────────
GREEK_ABBREVIATION_MAP: Dict[str, str] = {
    "κ.ά.": "και άλλα",
    "κ.λπ.": "και λοιπά",
    "κτλ.": "και τα λοιπά",
    "κ.α.": "και άλλα",
    "π.χ.": "παραδείγματος χάριν",
    "δηλ.": "δηλαδή",
    "π.Χ.": "προ Χριστού",
    "μ.Χ.": "μετά Χριστόν",
    "π.μ.": "προ μεσημβρίας",
    "μ.μ.": "μετά μεσημβρίαν",
    "χλμ.": "χιλιόμετρα",
    "τ.μ.": "τετραγωνικά μέτρα",
    "τ.χλμ.": "τετραγωνικά χιλιόμετρα",
    "κ.μ.": "κυβικά μέτρα",
    "εκ.": "εκατομμύρια",
    "δισ.": "δισεκατομμύρια",
    "χλγρ.": "χιλιόγραμμα",
    "γρ.": "γραμμάρια",
    "κ.": "κύριος",
    "κα.": "κυρία",
    "Δρ.": "Διδάκτωρ",
    "σελ.": "σελίδα",
    "άρθ.": "άρθρο",
    "παρ.": "παράγραφος",
    "αι.": "αιώνας",
    "βλ.": "βλέπε",
    "ό.π.": "όπου παραπάνω",
    "εκδ.": "έκδοση",
    "επιμ.": "επιμέλεια",
    "μτφρ.": "μετάφραση",
    "κ.ο.κ.": "και ούτω καθεξής",
    "λ.χ.": "λόγου χάριν",
}


# ── Number to Greek Words ─────────────────────────────────────────────────
# Units, teens, and hundreds inflect for grammatical gender in Greek
# (e.g. "τρεις γυναίκες" / "τρία βιβλία", "τετρακόσιες χιλιάδες").
UNITS = {
    0: "μηδέν", 1: "ένα", 2: "δύο", 3: "τρία", 4: "τέσσερα",
    5: "πέντε", 6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννέα"
}
UNITS_MASC = {0: "μηδέν", 1: "ένας", 2: "δύο", 3: "τρεις", 4: "τέσσερις",
              5: "πέντε", 6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννέα"}
UNITS_FEM = {0: "μηδέν", 1: "μία", 2: "δύο", 3: "τρεις", 4: "τέσσερις",
             5: "πέντε", 6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννέα"}

TEENS = {
    10: "δέκα", 11: "έντεκα", 12: "δώδεκα", 13: "δεκατρία",
    14: "δεκατέσσερα", 15: "δεκαπέντε", 16: "δεκαέξι",
    17: "δεκαεπτά", 18: "δεκαοκτώ", 19: "δεκαεννέα"
}
TEENS_MASC_FEM = {**TEENS, 13: "δεκατρείς", 14: "δεκατέσσερις"}

TENS = {
    2: "είκοσι", 3: "τριάντα", 4: "σαράντα", 5: "πενήντα",
    6: "εξήντα", 7: "εβδομήντα", 8: "ογδόντα", 9: "ενενήντα"
}

HUNDREDS = {
    1: "εκατό", 2: "διακόσια", 3: "τριακόσια", 4: "τετρακόσια",
    5: "πεντακόσια", 6: "εξακόσια", 7: "επτακόσια", 8: "οκτακόσια",
    9: "εννιακόσια"
}
HUNDREDS_MASC = {
    1: "εκατό", 2: "διακόσιοι", 3: "τριακόσιοι", 4: "τετρακόσιοι",
    5: "πεντακόσιοι", 6: "εξακόσιοι", 7: "επτακόσιοι", 8: "οκτακόσιοι",
    9: "εννιακόσιοι"
}
HUNDREDS_FEM = {
    1: "εκατό", 2: "διακόσιες", 3: "τριακόσιες", 4: "τετρακόσιες",
    5: "πεντακόσιες", 6: "εξακόσιες", 7: "επτακόσιες", 8: "οκτακόσιες",
    9: "εννιακόσιες"
}

THOUSAND_SINGULAR = {"masculine": "χίλιοι", "feminine": "χίλιες", "neuter": "χίλια"}

GREEK_MONTHS_MAP = {
    "Ιανουαρίου": "Ιανουάριος", "Φεβρουαρίου": "Φεβρουάριος",
    "Μαρτίου": "Μάρτιος", "Απριλίου": "Απρίλιος",
    "Μαΐου": "Μάιος", "Ιουνίου": "Ιούνιος",
    "Ιουλίου": "Ιούλιος", "Αυγούστου": "Αύγουστος",
    "Σεπτεμβρίου": "Σεπτέμβριος", "Οκτωβρίου": "Οκτώβριος",
    "Νοεμβρίου": "Νοέμβριος", "Δεκεμβρίου": "Δεκέμβριος",
}


def _normalize_gender(gender: str) -> str:
    return "neuter" if gender in ("neutral", "neuter") else gender


def _hundreds_group_to_greek(n: int, gender: str) -> str:
    """Convert 0-999 to Greek words in the requested gender."""
    units = {"masculine": UNITS_MASC, "feminine": UNITS_FEM}.get(gender, UNITS)
    teens = TEENS if gender == "neuter" else TEENS_MASC_FEM
    hundreds = {"masculine": HUNDREDS_MASC, "feminine": HUNDREDS_FEM}.get(
        gender, HUNDREDS
    )

    parts = []
    if n >= 100:
        word = hundreds[n // 100]
        n %= 100
        # "εκατόν" before a continuing number: "εκατόν είκοσι".
        if word == "εκατό" and n > 0:
            word = "εκατόν"
        parts.append(word)
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    elif n >= 10:
        parts.append(teens[n])
        n = 0
    if n > 0:
        parts.append(units[n])
    return " ".join(parts)


def num_to_greek(n: int, gender: str = "neutral") -> str:
    """Convert an integer to Greek words with grammatical gender agreement.

    ``gender`` applies to the final (units) group; the thousands group always
    agrees with the feminine "χιλιάδες", millions/billions are neuter.
    """
    gender = _normalize_gender(gender)
    if n == 0:
        return UNITS[0]
    if n < 0:
        return "μείον " + num_to_greek(abs(n), gender)

    parts = []

    # Millions and billions are needed for population and budget figures.
    for scale, singular, plural in (
        (1_000_000_000, "δισεκατομμύριο", "δισεκατομμύρια"),
        (1_000_000, "εκατομμύριο", "εκατομμύρια"),
    ):
        if n >= scale:
            count = n // scale
            if count == 1:
                parts.append(f"ένα {singular}")
            else:
                parts.append(num_to_greek(count, "neuter"))
                parts.append(plural)
            n %= scale

    # Thousands: "χιλιάδες" is feminine, so its count is feminine
    # ("τετρακόσιες ογδόντα δύο χιλιάδες").
    if n >= 1000:
        thousands = n // 1000
        if thousands == 1:
            parts.append(THOUSAND_SINGULAR[gender])
        else:
            parts.append(_hundreds_group_to_greek(thousands, "feminine"))
            parts.append("χιλιάδες")
        n %= 1000

    if n > 0:
        parts.append(_hundreds_group_to_greek(n, gender))

    return " ".join(parts)


ORDINAL_UNITS = {
    "masculine": {
        1: "πρώτος", 2: "δεύτερος", 3: "τρίτος", 4: "τέταρτος",
        5: "πέμπτος", 6: "έκτος", 7: "έβδομος", 8: "όγδοος", 9: "ένατος",
    },
    "feminine": {
        1: "πρώτη", 2: "δεύτερη", 3: "τρίτη", 4: "τέταρτη",
        5: "πέμπτη", 6: "έκτη", 7: "έβδομη", 8: "όγδοη", 9: "ένατη",
    },
    "neuter": {
        1: "πρώτο", 2: "δεύτερο", 3: "τρίτο", 4: "τέταρτο",
        5: "πέμπτο", 6: "έκτο", 7: "έβδομο", 8: "όγδοο", 9: "ένατο",
    },
}

ORDINAL_TENS = {
    "masculine": {
        10: "δέκατος", 20: "εικοστός", 30: "τριακοστός",
        40: "τεσσαρακοστός", 50: "πεντηκοστός", 60: "εξηκοστός",
        70: "εβδομηκοστός", 80: "ογδοηκοστός", 90: "ενενηκοστός",
    },
    "feminine": {
        10: "δέκατη", 20: "εικοστή", 30: "τριακοστή",
        40: "τεσσαρακοστή", 50: "πεντηκοστή", 60: "εξηκοστή",
        70: "εβδομηκοστή", 80: "ογδοηκοστή", 90: "ενενηκοστή",
    },
    "neuter": {
        10: "δέκατο", 20: "εικοστό", 30: "τριακοστό",
        40: "τεσσαρακοστό", 50: "πεντηκοστό", 60: "εξηκοστό",
        70: "εβδομηκοστό", 80: "ογδοηκοστό", 90: "ενενηκοστό",
    },
}


def ordinal_to_greek(n: int, gender: str = "neuter") -> str:
    """Convert common ordinal numbers to Greek words."""
    gender = _normalize_gender(gender)
    if n in ORDINAL_UNITS[gender]:
        return ORDINAL_UNITS[gender][n]
    if n in ORDINAL_TENS[gender]:
        return ORDINAL_TENS[gender][n]
    if 10 < n < 20:
        return f"{ORDINAL_TENS[gender][10]} {ORDINAL_UNITS[gender][n - 10]}"
    if 20 < n < 100:
        tens = (n // 10) * 10
        unit = n % 10
        return f"{ORDINAL_TENS[gender][tens]} {ORDINAL_UNITS[gender][unit]}"
    return num_to_greek(n, gender)


def year_to_greek(year: int) -> str:
    """Convert a year to Greek reading style.

    "χίλια οκτακόσια είκοσι ένα" for 1821, "δύο χιλιάδες είκοσι τρία" for 2023.
    """
    return num_to_greek(year, "neuter")


# ── Regex Patterns ────────────────────────────────────────────────────────

class GreekRegexNormalizer:
    """Regex-based Greek text normalizer for TTS preprocessing."""

    def __init__(self, expand_numbers: bool = True, expand_acronyms: bool = True,
                 expand_abbreviations: bool = True, expand_dates: bool = True,
                 expand_times: bool = True, expand_currencies: bool = True,
                 expand_percentages: bool = True):
        self.expand_numbers = expand_numbers
        self.expand_acronyms = expand_acronyms
        self.expand_abbreviations = expand_abbreviations
        self.expand_dates = expand_dates
        self.expand_times = expand_times
        self.expand_currencies = expand_currencies
        self.expand_percentages = expand_percentages

    def normalize(self, text: str) -> str:
        """Apply all enabled normalizations to the text."""
        result = text

        # Currency expressions must be handled before abbreviation expansion so
        # forms like "δισ. €" remain attached to their amount.
        if self.expand_currencies:
            result = self._expand_currencies(result)

        # Order matters: abbreviations contain periods
        if self.expand_abbreviations:
            result = self._expand_abbreviations(result)

        # Acronyms (before numbers so "AMKA" etc. don't get confused)
        if self.expand_acronyms:
            result = self._expand_acronyms(result)

        # Dates
        if self.expand_dates:
            result = self._expand_dates(result)

        # Times
        if self.expand_times:
            result = self._expand_times(result)

        # Percentages
        if self.expand_percentages:
            result = self._expand_percentages(result)

        if self.expand_numbers:
            # Long digit runs (phone numbers) before cardinals, so they are
            # read digit-by-digit instead of as millions.
            result = self._expand_phone_digits(result)

            # Simple fractions after dates (so 25/03/1821 is already consumed).
            result = self._expand_fractions(result)

            # Negative sign before decimals/cardinals.
            result = self._expand_minus(result)

            # Decimal numbers outside percentages/currencies ("5,3").
            result = self._expand_decimals(result)

        # Ordinal numbers (must be before cardinal)
        result = self._expand_ordinals(result)

        # Cardinal numbers (last, as they're the most general)
        if self.expand_numbers:
            result = self._expand_cardinals(result)

        return result

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known Greek abbreviations."""
        result = text
        for abbr, full in sorted(GREEK_ABBREVIATION_MAP.items(), key=lambda x: -len(x[0])):
            # Replace only when the abbreviation appears as a standalone token
            pattern = re.escape(abbr)
            result = re.sub(r'(?<!\w)' + pattern + r'(?!\w)', full, result)
        return result

    def _expand_acronyms(self, text: str) -> str:
        """Expand known Greek acronyms."""
        result = text
        for acronym, full in sorted(GREEK_ACRONYM_MAP.items(), key=lambda x: -len(x[0])):
            pattern = re.escape(acronym)
            result = re.sub(r'(?<!\w)' + pattern + r'(?!\w)', full, result)
        return result

    def _expand_dates(self, text: str) -> str:
        """Expand date patterns like '25η Μαρτίου 1821'."""
        # Pattern: DD Month YYYY (e.g., "25η Μαρτίου 1821")
        month_pattern = r'(Ιανουαρίου|Φεβρουαρίου|Μαρτίου|Απριλίου|Μαΐου|' \
                        r'Ιουνίου|Ιουλίου|Αυγούστου|Σεπτεμβρίου|Οκτωβρίου|' \
                        r'Νοεμβρίου|Δεκεμβρίου)'

        def _replace_date(m: re.Match) -> str:
            day_str = m.group(1)
            month = m.group(3)
            year_str = m.group(4)
            day_num = int(re.sub(r'\D', '', day_str))
            year_num = int(year_str)
            day_word = (
                "πρώτη" if day_num == 1 else num_to_greek(day_num, "feminine")
            )
            year_word = year_to_greek(year_num)
            return f"{day_word} {month} {year_word}"

        # "25η Μαρτίου 1821" or "25 Μαρτίου 1821"
        pattern = r'(\d+)(?:η|ος|ο)?\s+(' + month_pattern + r')\s+(\d{4})'
        text = re.sub(pattern, _replace_date, text)

        # Day + month without a year: "στις 3 Φεβρουαρίου". Days are feminine
        # (ημέρα); the 1st is read as an ordinal ("πρώτη Μαρτίου").
        def _replace_day_month(m: re.Match) -> str:
            day_num = int(m.group(1))
            day_word = (
                "πρώτη" if day_num == 1 else num_to_greek(day_num, "feminine")
            )
            return f"{day_word} {m.group(2)}"

        text = re.sub(
            r'(\d{1,2})(?:η|ης)?\s+(' + month_pattern + r')',
            _replace_day_month,
            text,
        )

        # DD/MM/YYYY or DD-MM-YYYY
        def _replace_numeric_date(m: re.Match) -> str:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            months = ["Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου",
                      "Μαΐου", "Ιουνίου", "Ιουλίου", "Αυγούστου",
                      "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"]
            month_name = months[mo - 1] if 1 <= mo <= 12 else str(mo)
            return f"{num_to_greek(d, 'feminine')} {month_name} {year_to_greek(y)}"

        text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', _replace_numeric_date, text)
        text = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', _replace_numeric_date, text)

        return text

    def _expand_times(self, text: str) -> str:
        """Expand time patterns like '14:30' or '09:00'."""
        def _replace_time(match: re.Match) -> str:
            hours, minutes = int(match.group(1)), int(match.group(2))
            hour_word = num_to_greek(hours, "feminine")
            if minutes == 0:
                return f"{hour_word} ακριβώς"
            elif minutes == 30:
                return f"{hour_word} και μισή"
            else:
                min_word = num_to_greek(minutes, "neuter")
                return f"{hour_word} και {min_word}"

        text = re.sub(r'(\d{1,2}):(\d{2})(?!\d)', _replace_time, text)
        return text

    def _expand_currencies(self, text: str) -> str:
        """Expand currency patterns like '467,590 δισ. €'"""
        currency_names = {
            "€": "ευρώ",
            "$": "δολάρια",
            "£": "λίρες",
        }

        def _format_scaled_amount(raw_amount: str, suffix: str,
                                  currency: str) -> str:
            suffix_word = "δισεκατομμύρια" if suffix.startswith("δισ") else "εκατομμύρια"
            minor_suffix = "εκατομμύρια" if suffix.startswith("δισ") else "χιλιάδες"
            minor_gender = "neuter" if minor_suffix == "εκατομμύρια" else "feminine"

            if "," in raw_amount:
                whole_part, decimal_part = raw_amount.split(",", 1)
                whole = int(whole_part.replace(".", ""))
                # The decimal part counts thousandths of the scale:
                # "467,5 δισ." is 467 billion 500 million, not 5 million.
                decimal_part = decimal_part[:3]
                decimal = int(decimal_part) * 10 ** (3 - len(decimal_part))
                parts = [num_to_greek(whole, "neuter"), suffix_word]
                if decimal:
                    parts.extend(
                        [num_to_greek(decimal, minor_gender), minor_suffix]
                    )
                parts.append(currency)
                return " ".join(parts)

            whole = int(raw_amount.replace(".", ""))
            return f"{num_to_greek(whole, 'neuter')} {suffix_word} {currency}"

        def _format_simple_amount(raw_amount: str, currency: str) -> str:
            if "," in raw_amount:
                whole_part, cents_part = raw_amount.split(",", 1)
                whole = int(whole_part.replace(".", ""))
                cents = int(cents_part)
                if cents:
                    return (f"{num_to_greek(whole, 'neuter')} {currency} και "
                            f"{num_to_greek(cents, 'neuter')} λεπτά")
                return f"{num_to_greek(whole, 'neuter')} {currency}"

            whole = int(raw_amount.replace(".", ""))
            return f"{num_to_greek(whole, 'neuter')} {currency}"

        def _replace_currency(m: re.Match) -> str:
            amount = m.group(1)
            suffix = m.group(2) or ""
            currency = currency_names.get(m.group(3), "ευρώ")
            try:
                if suffix:
                    return _format_scaled_amount(amount, suffix, currency)
                return _format_simple_amount(amount, currency)
            except ValueError:
                return m.group(0)

        # "467,590 δισ. €" pattern
        text = re.sub(r'(\d[\d.]*(?:,\d+)?)\s*(δισ\.?|εκ\.?)?\s*([€$£])',
                      _replace_currency, text)

        # "$350.000" pattern
        def _replace_prefixed_currency(m: re.Match) -> str:
            currency = currency_names.get(m.group(1), "ευρώ")
            try:
                return _format_simple_amount(m.group(2), currency)
            except ValueError:
                return m.group(0)

        text = re.sub(r'([€$£])\s*(\d[\d.]*(?:,\d+)?)',
                      _replace_prefixed_currency, text)

        return text

    def _expand_percentages(self, text: str) -> str:
        """Expand percentage patterns like '12,5%'"""
        def _replace_pct(m: re.Match) -> str:
            num_str = m.group(1).replace(",", ".")
            try:
                val = float(num_str)
                if '.' in num_str or ',' in str(m.group(1)):
                    # Decimal percentage
                    int_part = int(val)
                    dec_part = num_str.split("." if "." in num_str else ",")[1]
                    return f"{num_to_greek(int_part, 'neuter')} και {num_to_greek(int(dec_part), 'neuter')} τοις εκατό"
                else:
                    return f"{num_to_greek(int(val), 'neuter')} τοις εκατό"
            except ValueError:
                return m.group(0)

        text = re.sub(r'([\d,.]+)\s*%', _replace_pct, text)
        return text

    def _expand_phone_digits(self, text: str) -> str:
        """Read long digit runs (phone numbers) digit-by-digit."""
        def _replace_digits(m: re.Match) -> str:
            return " ".join(UNITS[int(d)] for d in m.group(1))

        # 7+ contiguous digits are phone-number-like, never a quantity. A
        # sentence-final period may follow; only digit continuations are out.
        text = re.sub(r'(?<![\d.,])(\d{7,})(?!\d)(?![.,]\d)', _replace_digits, text)
        # International prefix: "+30" -> "συν τριάντα".
        text = re.sub(r'\+(?=\d)', 'συν ', text)
        return text

    def _expand_fractions(self, text: str) -> str:
        """Expand simple proper fractions like '3/4'."""
        denominators = {
            2: ("δεύτερο", "δεύτερα"), 3: ("τρίτο", "τρίτα"),
            4: ("τέταρτο", "τέταρτα"), 5: ("πέμπτο", "πέμπτα"),
            6: ("έκτο", "έκτα"), 7: ("έβδομο", "έβδομα"),
            8: ("όγδοο", "όγδοα"), 9: ("ένατο", "ένατα"),
            10: ("δέκατο", "δέκατα"),
        }

        def _replace_fraction(m: re.Match) -> str:
            numerator, denominator = int(m.group(1)), int(m.group(2))
            if denominator not in denominators or numerator >= denominator:
                return m.group(0)
            singular, plural = denominators[denominator]
            word = singular if numerator == 1 else plural
            return f"{num_to_greek(numerator, 'neuter')} {word}"

        # Dates (25/03/1821) were already expanded; remaining d/d with a
        # proper-fraction shape are fractions. Day/month leftovers like 25/03
        # fail the numerator < denominator check and stay untouched.
        return re.sub(
            r'(?<![\w/.,])(\d{1,2})/(\d{1,2})(?![\w/.,])', _replace_fraction, text
        )

    def _expand_minus(self, text: str) -> str:
        """Expand a leading negative sign before a number."""
        # Not after a digit or word character, so ranges like "45-67" survive.
        return re.sub(r'(?<![\w\d])-(?=\d)', 'μείον ', text)

    def _expand_decimals(self, text: str) -> str:
        """Expand decimal-comma numbers like '5,3' to 'πέντε κόμμα τρία'."""
        def _replace_decimal(m: re.Match) -> str:
            whole, decimal_part = m.group(1), m.group(2)
            whole_word = num_to_greek(int(whole), "neuter")
            if decimal_part.startswith("0"):
                decimal_word = " ".join(UNITS[int(d)] for d in decimal_part)
            else:
                decimal_word = num_to_greek(int(decimal_part), "neuter")
            return f"{whole_word} κόμμα {decimal_word}"

        return re.sub(
            r'(?<![\d.,])(\d+),(\d{1,3})(?!\d)(?![.,]\d)', _replace_decimal, text
        )

    def _expand_ordinals(self, text: str) -> str:
        """Expand ordinal numbers like '95η', '2ος', '3ο'"""
        GENDER_SUFFIX = {
            'η': 'feminine', 'ος': 'masculine', 'ο': 'neuter',
            'ης': 'feminine', 'ες': 'feminine', 'οι': 'masculine',
            'α': 'neuter', 'ων': 'neuter',
        }

        def _replace_ordinal(m: re.Match) -> str:
            num = int(m.group(1))
            suffix = m.group(2)
            gender = GENDER_SUFFIX.get(suffix, 'neuter')
            return ordinal_to_greek(num, gender)

        text = re.sub(r'(\d+)(ος|η|ο|ης|ες|οι|α|ων)\b', _replace_ordinal, text)
        return text

    def _expand_cardinals(self, text: str) -> str:
        """Expand cardinal numbers (standalone integers)."""
        def _replace_number(m: re.Match) -> str:
            num_str = m.group(1).replace(".", "")
            try:
                num = int(num_str)
                return num_to_greek(num, "neutral")
            except ValueError:
                return m.group(0)

        # Match standalone integers, including dot-separated Greek thousands.
        text = re.sub(r'(?<![\w\d.,])(\d+(?:\.\d+)*)(?![\w\d]|[.,]\d)',
                      _replace_number, text)

        return text


# ── Utility Functions ─────────────────────────────────────────────────────

def normalize_dataset(input_json_path: str, output_json_path: str,
                      normalizer: GreekRegexNormalizer = None) -> None:
    """Load a dataset JSON, normalize all sentences, and save."""
    import json

    if normalizer is None:
        normalizer = GreekRegexNormalizer()

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sentence in data['sentences']:
        sentence['original_text'] = sentence['text']
        sentence['normalized_text'] = normalizer.normalize(sentence['text'])

    data['preprocessing'] = {
        'method': 'regex',
        'normalizer': 'GreekRegexNormalizer',
        'settings': {
            'expand_numbers': normalizer.expand_numbers,
            'expand_acronyms': normalizer.expand_acronyms,
            'expand_abbreviations': normalizer.expand_abbreviations,
            'expand_dates': normalizer.expand_dates,
            'expand_times': normalizer.expand_times,
            'expand_currencies': normalizer.expand_currencies,
            'expand_percentages': normalizer.expand_percentages,
        }
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Normalized {len(data['sentences'])} sentences from {input_json_path}")
    print(f"Output written to {output_json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    normalizer = GreekRegexNormalizer()

    # Test examples
    test_sentences = [
        "Ο πληθυσμός ανέρχεται σε 10.482.487 κατοίκους σύμφωνα με την ΕΛΣΤΑΤ.",
        "Η ανεξαρτησία κηρύχθηκε την 25η Μαρτίου 1821.",
        "Το ποσοστό ανεργίας μειώθηκε στο 12,5% το 2023.",
        "Ο προϋπολογισμός ανήλθε στα 467,590 δισ. €.",
        "Η Ελλάδα είναι μέλος του ΝΑΤΟ από το 1952 και ιδρυτικό μέλος του ΟΗΕ.",
        "Η ΕΕ χρηματοδότησε το έργο μέσω του ΕΣΠΑ.",
        "Το ωράριο είναι από τις 09:00 έως τις 17:30.",
        "Η θερμοκρασία έπεσε στους -5°C.",
        "Κατέχει την 95η θέση στην κατάταξη των χωρών.",
        "Βλ. επίσης Μπίλλης (2025), σελ. 45-67, κ.ά.",
    ]

    if len(sys.argv) > 1:
        # File mode
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_normalized.json')
        normalize_dataset(input_file, output_file, normalizer)
    else:
        # Demo mode
        print("=" * 70)
        print("GreekRegexNormalizer - Demo")
        print("=" * 70)
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n--- Sentence {i} ---")
            print(f"INPUT:  {sentence}")
            print(f"OUTPUT: {normalizer.normalize(sentence)}")
        print("\n" + "=" * 70)
