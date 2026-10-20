"""
Romanizer module for Ethiopian scripts (Ge'ez/Ethiopic).
Meta MMS-TTS models for Amharic (amh) and Tigrinya (tir) require romanized text input.

Primary strategy: USC/ISI uroman library — the same tool used to prepare the MMS training data.
- Produces phonemically accurate output (6th-order Ge'ez syllables get short schwa vowels).
- Double-vowel normalization applied: 'aa'→'a', 'ee'→'e', 'oo'→'o', 'hh'→'h'.

Fallback: GEEZ_MAP character-level transliteration (only when uroman is unavailable).
WARNING: GEEZ_MAP is imprecise for 6th-order (vowel-less) syllables such as ምን, ነው, ይህ, ልብ —
it maps them to bare consonant clusters (mn, naw, yh, lb) which the MMS model was never
trained on, causing slurring and mispronunciation. Prefer uroman whenever possible.
"""

import re
import logging

logger = logging.getLogger("ethiopian_tts.romanizer")

# ---------------------------------------------------------------------------
# uroman singleton — lazy-loaded once, cached for lifetime of process
# ---------------------------------------------------------------------------
_uroman_instance = None

def _get_uroman():
    """Return a cached Uroman instance, or None if uroman is not installed."""
    global _uroman_instance
    if _uroman_instance is not None:
        return _uroman_instance
    try:
        import uroman as ur_module
        _uroman_instance = ur_module.Uroman()
        logger.info("uroman loaded successfully — using uroman pipeline for Ge'ez romanization")
    except Exception as exc:
        logger.warning(f"uroman unavailable ({exc}); falling back to GEEZ_MAP")
        _uroman_instance = False  # sentinel so we don't retry every call
    return _uroman_instance if _uroman_instance else None


def _romanize_geez_uroman(text: str):
    """
    Romanize Ge'ez text using uroman with double-vowel normalization.
    Produces the same token distribution as the MMS training corpus.
    Returns None if uroman is unavailable (caller should use GEEZ_MAP fallback).
    """
    u = _get_uroman()
    if u is None:
        return None  # signal caller to use GEEZ_MAP fallback

    raw = u.romanize_string(text).lower()
    # Keep only ASCII letters, apostrophe, space; drop numbers/punctuation
    raw = re.sub(r"[^a-z' ]", " ", raw)
    # Normalize doubled vowels produced by uroman (uroman sometimes generates
    # 'salaame'→'salame', 'dahenaa'→'dahena') — collapse aa/ee/oo/hh
    raw = re.sub(r"aa", "a", raw)
    raw = re.sub(r"ee", "e", raw)
    raw = re.sub(r"oo", "o", raw)
    raw = re.sub(r"hh", "h", raw)
    # Collapse multiple spaces
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


# Comprehensive Ge'ez (Ethiopic) to Roman transliteration table
# Authentic phonetic mapping for Meta MMS-TTS (amh & tir)
GEEZ_MAP = {
    # Ha
    "ሀ": "ha", "ሁ": "hu", "ሂ": "hi", "ሃ": "ha", "ሄ": "he", "ህ": "h", "ሆ": "ho",
    # La
    "ለ": "la", "ሉ": "lu", "ሊ": "li", "ላ": "la", "ሌ": "le", "ል": "l", "ሎ": "lo", "ሏ": "lwa",
    # Hha
    "ሐ": "ha", "ሑ": "hu", "ሒ": "hi", "ሓ": "ha", "ሔ": "he", "ሕ": "h", "ሖ": "ho", "ሗ": "hwa",
    # Ma
    "መ": "ma", "ሙ": "mu", "ሚ": "mi", "ማ": "ma", "ሜ": "me", "ም": "m", "ሞ": "mo", "ሟ": "mwa", "ፙ": "mya",
    # Ssa
    "ሠ": "sa", "ሡ": "su", "ሢ": "si", "ሣ": "sa", "ሤ": "se", "ሥ": "s", "ሦ": "so", "ሧ": "swa",
    # Ra
    "ረ": "ra", "ሩ": "ru", "ሪ": "ri", "ራ": "ra", "ሬ": "re", "ር": "r", "ሮ": "ro", "ሯ": "rwa", "ፘ": "rya",
    # Sa
    "ሰ": "sa", "ሱ": "su", "ሲ": "si", "ሳ": "sa", "ሴ": "se", "ስ": "s", "ሶ": "so", "ሷ": "swa",
    # Sha
    "ሸ": "sha", "ሹ": "shu", "ሺ": "shi", "ሻ": "sha", "ሼ": "she", "ሽ": "sh", "ሾ": "sho", "shared": "shwa", "ሿ": "shwa",
    # Qa
    "ቀ": "qa", "ቁ": "qu", "ቂ": "qi", "ቃ": "qa", "ቄ": "qe", "ቅ": "q", "ቆ": "qo", "ቈ": "qwa", "ቊ": "qwi", "ቋ": "qwa", "ቌ": "qwe", "ቍ": "qw",
    # Qha (Tigrinya)
    "ቐ": "qha", "ቑ": "qhu", "ቒ": "qhi", "ቓ": "qha", "ቔ": "qhe", "ቕ": "qh", "ቖ": "qho", "ቘ": "qhwa", "ቚ": "qhwi", "ቛ": "qhwa", "ቜ": "qhwe", "ቝ": "qhw",
    # Ba
    "በ": "ba", "ቡ": "bu", "ቢ": "bi", "ባ": "ba", "ቤ": "be", "ብ": "b", "ቦ": "bo", "ቧ": "bwa",
    # Ta
    "ተ": "ta", "ቱ": "tu", "ቲ": "ti", "ታ": "ta", "ቴ": "te", "ት": "t", "ቶ": "to", "ቷ": "twa",
    # Cha
    "ቸ": "cha", "ቹ": "chu", "ቺ": "chi", "ቻ": "cha", "ቼ": "che", "ች": "ch", "ቾ": "cho", "ቿ": "chwa",
    # Hha (Alternate)
    "ኀ": "ha", "ኁ": "hu", "ኂ": "hi", "ኃ": "ha", "ኄ": "he", "ኅ": "h", "ኆ": "ho", "ኈ": "hwa", "ኊ": "hwi", "ኋ": "hwa", "ኌ": "hwe", "ኍ": "hw",
    # Na
    "ነ": "na", "ኑ": "nu", "ኒ": "ni", "ና": "na", "ኔ": "ne", "ን": "n", "ኖ": "no", "ኗ": "nwa",
    # Nya
    "ኘ": "nya", "ኙ": "nyu", "ኚ": "nyi", "ኛ": "nya", "ኜ": "nye", "ኝ": "ny", "ኞ": "nyo", "ኟ": "nywa",
    # 'A (Alef)
    "አ": "a", "ኡ": "u", "ኢ": "i", "ኣ": "a", "ኤ": "e", "እ": "e", "ኦ": "o", "ኧ": "wa",
    # Ka
    "ከ": "ka", "ኩ": "ku", "ኪ": "ki", "ካ": "ka", "ኬ": "ke", "ክ": "k", "ኮ": "ko", "ኰ": "kwa", "ኲ": "kwi", "ኳ": "kwa", "ኴ": "kwe", "ኵ": "kw",
    # Kha (Tigrinya)
    "ኸ": "kha", "ኹ": "khu", "ኺ": "khi", "ኻ": "kha", "ኼ": "khe", "ኽ": "kh", "ኾ": "kho", "ዀ": "khwa", "ዂ": "khwi", "ዃ": "khwa", "ዄ": "khwe", "ዅ": "khw",
    # Wa
    "ወ": "wa", "ዉ": "wu", "ዊ": "wi", "ዋ": "wa", "ዌ": "we", "ው": "w", "ዎ": "wo",
    # 'A (Ayn)
    "ዐ": "a", "ዑ": "u", "ዒ": "i", "ዓ": "a", "ዔ": "e", "ዕ": "e", "ዖ": "o",
    # Za
    "ዘ": "za", "ዙ": "zu", "ዚ": "zi", "ዛ": "za", "ዜ": "ze", "ዝ": "z", "ዞ": "zo", "ዟ": "zwa",
    # Zha
    "ዠ": "zha", "ዡ": "zhu", "ዢ": "zhi", "ዣ": "zha", "ዤ": "zhe", "ዥ": "zh", "ዦ": "zho", "ዧ": "zhwa",
    # Ya
    "የ": "ya", "ዩ": "yu", "ዪ": "yi", "ያ": "ya", "ዬ": "ye", "ይ": "y", "ዮ": "yo", "ዯ": "ywa",
    # Da
    "ደ": "da", "ዱ": "du", "ዲ": "di", "ዳ": "da", "ዴ": "de", "ድ": "d", "ዶ": "do", "ዷ": "dwa",
    # Dha/Jha
    "ዸ": "dda", "ዹ": "ddu", "ዺ": "ddi", "ዻ": "dda", "ዼ": "dde", "ዽ": "dd", "ዾ": "ddo", "ዿ": "ddwa",
    # Ja
    "ጀ": "ja", "ጁ": "ju", "ጂ": "ji", "ጃ": "ja", "ጄ": "je", "ጅ": "j", "ጆ": "jo", "ጇ": "jwa",
    # Ga
    "ገ": "ga", "ጉ": "gu", "ጊ": "gi", "ጋ": "ga", "ጌ": "ge", "ግ": "g", "ጎ": "go", "ጐ": "gwa", "ጒ": "gwi", "ጓ": "gwa", "ጔ": "gwe", "ጕ": "gw",
    # Tta
    "ጠ": "ta", "ጡ": "tu", "ጢ": "ti", "ጣ": "ta", "ጤ": "te", "ጥ": "t", "ጦ": "to", "ጧ": "twa",
    # Ccha
    "ጨ": "cha", "ጩ": "chu", "ጪ": "chi", "ጫ": "cha", "ጬ": "che", "ጭ": "ch", "ጮ": "cho", "ጯ": "chwa",
    # Ppa
    "ጰ": "pa", "ጱ": "pu", "ጲ": "pi", "ጳ": "pa", "ጴ": "pe", "ጵ": "p", "ጶ": "po", "ጷ": "pwa",
    # Tsa
    "ጸ": "tsa", "ጹ": "tsu", "ጺ": "tsi", "ጻ": "tsa", "ጼ": "tse", "ጽ": "ts", "ጾ": "tso", "ጿ": "tswa",
    # Tza
    "ፀ": "tsa", "ፁ": "tsu", "ጺ": "tsi", "ፃ": "tsa", "ፄ": "tse", "ፅ": "ts", "ፆ": "tso",
    # Fa
    "ፈ": "fa", "ፉ": "fu", "ፊ": "fi", "ፋ": "fa", "ፌ": "fe", "ፍ": "f", "ፎ": "fo", "ፏ": "fwa",
    # Pa
    "ፐ": "pa", "ፑ": "pu", "ፒ": "pi", "ፓ": "pa", "ፔ": "pe", "ፕ": "p", "ፖ": "po", "ፗ": "pwa",
    # Numerals
    "፩": "1", "፪": "2", "፫": "3", "፬": "4", "፭": "5", "፮": "6", "፯": "7", "፰": "8", "፱": "9", "፲": "10",
    # Punctuation
    "፡": " ", "።": ". ", "፣": ", ", "፤": "; ", "፥": ": ", "፦": ": ", "፧": "? ", "፨": " "
}

def romanize_text(text: str, language_id: str = "amh") -> str:
    """
    Romanize input text for Meta MMS speech synthesis.

    - Amharic (amh) & Tigrinya (tir): Ge'ez script → romanized via uroman (primary) or GEEZ_MAP (fallback).
    - Oromo (orm) & Somali (som): already Latin-script, normalized only.

    uroman is the USC/ISI romanization tool used to prepare the MMS training corpus.
    Using it here ensures the tokenizer input matches the training distribution exactly,
    giving correct duration prediction and clean phoneme alignment.
    """
    if language_id in ("orm", "som"):
        t = text.lower()
        t = re.sub(r"[?!,.:;\"()\[\]—–\-_/]+", " ", t)
        t = re.sub(r"[^a-z' ]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    # For Ge'ez scripts (amh, tir): try uroman first
    uroman_result = _romanize_geez_uroman(text)
    if uroman_result is not None:
        logger.debug(f"uroman [{language_id}]: '{text[:40]}' → '{uroman_result[:60]}'")
        return uroman_result

    # Fallback: character-level GEEZ_MAP
    logger.warning(f"Using GEEZ_MAP fallback for {language_id} romanization")
    result = []
    for char in text:
        if char in GEEZ_MAP:
            result.append(GEEZ_MAP[char])
        else:
            result.append(char)

    romanized = "".join(result)
    # Clean non-alphabetic characters that are not spaces or apostrophes
    romanized = re.sub(r"[^a-zA-Z' \n]", " ", romanized)
    # Collapse multiple spaces and lowercase
    romanized = re.sub(r"\s+", " ", romanized).strip().lower()
    return romanized


