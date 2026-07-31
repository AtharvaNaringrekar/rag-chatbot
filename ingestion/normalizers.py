import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """
    Standardize whitespaces in the text.
    Replaces carriage returns (\r\n) with newlines (\n), replaces multiple consecutive 
    spaces with a single space, and replaces multiple consecutive newlines (more than two) 
    with exactly two newlines (preserving paragraph separations).

    Args:
        text: The raw text string to normalize.

    Returns:
        The normalized text string.
    """
    if not text:
        return ""
    
    # 1. Normalize line endings to Unix style (\n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Replace three or more newlines with exactly two (to maintain paragraph division)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 3. Replace multiple spaces/tabs within a single line with a single space
    # We split by line to avoid collapsing newlines into spaces
    lines = []
    for line in text.split("\n"):
        # Strip trailing and leading whitespace for each line
        cleaned_line = line.strip()
        # Collapse multiple horizontal spaces into a single space
        cleaned_line = re.sub(r"[ \t]+", " ", cleaned_line)
        lines.append(cleaned_line)
        
    return "\n".join(lines)


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters from text, except standard newlines and tabs,
    which are crucial for code blocks and text structure.

    Args:
        text: The text to clean.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    
    return "".join(
        char for char in text 
        if unicodedata.category(char)[0] != "C" or char in ("\n", "\t")
    )


def clean_boilerplate(text: str) -> str:
    """
    Identifies and removes website navigation menus, social media links, 
    copyright notices, contact details, and generic footer/header boilerplate.
    Preserves all technical documentation, code blocks, and tables.
    """
    if not text:
        return ""
        
    lines = text.split("\n")
    cleaned_lines = []
    lines_removed = 0
    chars_removed = 0
    
    in_code_block = False
    
    # Precompile regex matching patterns
    email_regex = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    phone_regex = re.compile(r"\+?[0-9]{1,4}[-.\s\(\)]+[0-9]{3,4}[-.\s\(\)]+[0-9]{3,4}[-.\s\(\)]+[0-9]{3,4}")
    
    copyright_regex = re.compile(r"(?i)copyright\s+(\(c\)|©|\d)|all\s+rights\s+reserved|©\s*\d{4}")
    social_regex = re.compile(r"(?i)(follow\s+us\s+on|facebook\.com|twitter\.com|linkedin\.com|instagram\.com|youtube\.com)")
    
    # Key boilerplate phrases to match case-insensitively
    boilerplate_phrases = {
        "privacy policy", "terms of service", "terms of use", "terms and conditions", "terms & conditions",
        "cookie policy", "cookie notice", "cookie preferences", "cookie settings",
        "get a quote", "contact us", "contact sales", "support portal", "tech support", "request a demo",
        "whitepapers", "webinars", "press release", "press", "data sheets", "careers", "about us",
        "back to top", "next page", "previous page", "share this", "all rights reserved"
    }
    
    for line in lines:
        stripped = line.strip()
        
        # Keep code blocks entirely intact
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue
            
        if in_code_block:
            cleaned_lines.append(line)
            continue
            
        if not stripped:
            cleaned_lines.append(line)
            continue
            
        # 1. Contact details filtering
        temp_line = line
        if email_regex.search(temp_line):
            temp_line = email_regex.sub("", temp_line).strip()
        if phone_regex.search(temp_line):
            temp_line = phone_regex.sub("", temp_line).strip()
            
        if not temp_line and line:
            lines_removed += 1
            chars_removed += len(line)
            continue
            
        # 2. Copyright filtering
        if copyright_regex.search(stripped):
            lines_removed += 1
            chars_removed += len(line)
            continue
            
        # 3. Social media links filtering
        if social_regex.search(stripped):
            lines_removed += 1
            chars_removed += len(line)
            continue
            
        # 4. Standalone boilerplate headers or links
        stripped_lower = stripped.lower().strip(" \t.®™|-•")
        if stripped_lower in boilerplate_phrases:
            lines_removed += 1
            chars_removed += len(line)
            continue
            
        # 5. Delimited navigation bars/menus
        if "|" in stripped or "•" in stripped or "·" in stripped or "  " in stripped:
            parts = [p.strip().lower().strip(" \t.®™|-•") for p in re.split(r"[|•·]|\s{2,}", stripped)]
            bp_count = sum(1 for p in parts if p in boilerplate_phrases or p in ["home", "about", "products", "solutions", "pricing", "resources", "blog", "developers", "login", "sign up", "download"])
            if len(parts) > 1 and bp_count / len(parts) >= 0.5:
                lines_removed += 1
                chars_removed += len(line)
                continue
                
        # 6. Corporate address detection
        address_indicators = ["st.", "street", "ave.", "avenue", "suite", "floor", "building", "bldg", "road", "rd.", "drive", "dr.", "highway", "hwy", "p.o. box", "po box", "zip code", "postal code"]
        has_zip = re.search(r"\b\d{5}(-\d{4})?\b", stripped)
        has_indicator = any(ind in stripped.lower() for ind in address_indicators)
        is_company_addr = (has_zip and has_indicator) or (has_zip and any(co in stripped.lower() for co in ["inc.", "ltd.", "llc.", "corporation", "corp."]))
        if is_company_addr:
            lines_removed += 1
            chars_removed += len(line)
            continue

        cleaned_lines.append(temp_line)
        
    cleaned_text = "\n".join(cleaned_lines)
    
    if lines_removed > 0 or chars_removed > 0:
        import logging
        normalizer_logger = logging.getLogger(__name__)
        normalizer_logger.info(
            f"[BOILERPLATE CLEANER DIAGNOSTIC] Lines removed: {lines_removed} | "
            f"Characters removed: {chars_removed} | "
            f"Final cleaned document size: {len(cleaned_text)} characters"
        )
        
    return cleaned_text


def normalize_ligatures(text: str) -> str:
    """
    Normalizes typographic ligatures and malformed characters extracted from PDFs
    to their standard multi-character equivalents.
    """
    if not text:
        return ""
    
    # Mapping of unicode ligatures/malformed characters to ASCII equivalents
    ligature_map = {
        "\u019f": "ti",  # LATIN CAPITAL LETTER O WITH MIDDLE TILDE (Ɵ) -> ti
        "\u012c": "ti",  # LATIN CAPITAL LETTER I WITH BREVE (Ĭ) -> ti
        "\u01ac": "ti",  # LATIN SMALL LETTER I WITH BREVE (ĭ) -> ti
        "\ufb01": "fi",  # LATIN SMALL LIGATURE FI (ﬁ) -> fi
        "\ufb02": "fl",  # LATIN SMALL LIGATURE FL (ﬂ) -> fl
        "\ufb03": "ffi", # LATIN SMALL LIGATURE FFI (ﬃ) -> ffi
        "\ufb04": "ffl", # LATIN SMALL LIGATURE FFL (ﬄ) -> ffl
        "\ufb00": "ff",  # LATIN SMALL LIGATURE FF (ﬀ) -> ff
        "\u014c": "ft",  # LATIN CAPITAL LETTER O WITH MACRON (Ō) -> ft
        "\u019e": "ft",  # LATIN SMALL LETTER N WITH LONG RIGHT LEG (ƞ) -> ft
        "\u01a9": "tt",  # LATIN CAPITAL LETTER ESH (Ʃ) -> tt
        "\u01ab": "tt",  # LATIN SMALL LETTER T WITH PALATAL HOOK (ƫ) -> tt
        "\ufb05": "st",  # LATIN SMALL LIGATURE LONG S T (ﬅ) -> st
        "\ufb06": "st",  # LATIN SMALL LIGATURE ST (ﬆ) -> st
    }
    
    # Fast translation table lookup
    translation_table = str.maketrans(ligature_map)
    return text.translate(translation_table)


def normalize_text(text: str) -> str:
    """
    Run complete text sanitization sequence.
    Normalizes ligatures, removes control characters, filters out website boilerplate, 
    standardizes whitespaces, and strips outer whitespace boundaries.

    Args:
        text: Raw text content.

    Returns:
        Sanitized and normalized text content.
    """
    if not text:
        return ""
    
    text = normalize_ligatures(text)
    text = remove_control_characters(text)
    text = clean_boilerplate(text)
    text = normalize_whitespace(text)
    return text.strip()
