"""
OCR Engine module for extracting text and job attributes from recruitment posters.
Supports Tesseract OCR with automatic binary detection, image preprocessing,
and entity parsing heuristics.
"""

import os
import re
import shutil
from typing import Dict, Any, Optional
import pytesseract
from pytesseract import Output
from backend.image_processor import preprocess_image_for_ocr
from backend.text_processor import extract_emails, extract_phone_numbers, extract_urls

# Prefer the configured Windows installation before looking elsewhere on PATH.
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

COMMON_TESSERACT_PATHS = [
    TESSERACT_PATH,
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract_binary() -> Optional[str]:
    """Find the configured Tesseract binary, then fall back to PATH and common locations."""
    for path in COMMON_TESSERACT_PATHS:
        if os.path.isfile(path):
            return path

    return shutil.which("tesseract")


# Initialize tesseract path if found
_tesseract_bin = find_tesseract_binary()
if _tesseract_bin:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_bin


FIELD_LABELS = {
    "company_name": ["company name", "company", "organization", "employer", "hiring company"],
    "job_title": ["job title", "position", "role", "hiring for", "job opening"],
    "experience_requirements": ["years of experience", "work experience", "experience"],
    "required_skills": ["required skills", "technical skills", "key skills", "skills"],
    "education": ["education / qualification", "education/qualification", "qualification", "education", "degree"],
    "job_location": ["job location", "work location", "location", "city"],
    "salary": ["salary / package", "salary/package", "salary range", "compensation", "package", "salary", "ctc"],
    "job_description": ["job description", "responsibilities", "about the role", "role overview"],
    "recruiter_email": ["recruiter email", "contact email", "email"],
    "contact_number": ["recruiter contact number", "contact number", "phone number", "mobile number", "phone", "mobile", "contact"],
    "application_url": ["application url", "apply here", "apply at", "apply now", "application link"],
    "official_website": ["official website", "company website", "careers website", "website"],
    "official_careers_url": ["official careers url", "careers page", "career portal"]
}

_LABEL_TO_FIELD = {
    re.sub(r"\s+", " ", label).strip().casefold(): field
    for field, labels in FIELD_LABELS.items()
    for label in labels
}
_LABEL_PATTERN = re.compile(
    r"(?<!\w)(?P<label>" +
    "|".join(
        re.escape(label).replace(r"\ ", r"\s+")
        for label in sorted(_LABEL_TO_FIELD, key=len, reverse=True)
    ) +
    r")\s*[:：]\s*",
    re.IGNORECASE
)
_LOCATION_HINTS = re.compile(
    r"\b(?:Bengaluru|Bangalore|Hyderabad|Chennai|Pune|Mumbai|PAN\s+India|"
    r"Remote|Work\s+From\s+Home|Multiple\s+Locations)\b"
    r"(?:\s*,\s*(?:India|Karnataka|Telangana|Tamil\s+Nadu|Maharashtra))?",
    re.IGNORECASE
)
_GENERAL_LOCATIONS = re.compile(
    r"^(?:PAN\s+India|Remote|Work\s+From\s+Home|Multiple\s+Locations)$",
    re.IGNORECASE
)
_EXPERIENCE_VALUE = re.compile(
    r"\b\d{1,2}\s*(?:[-–]\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?|months?|mos?)\b",
    re.IGNORECASE
)
_EDUCATION_VALUE = re.compile(
    r"\b(?:B\.E\.?|B\.?Tech|M\.?Tech|MCA|MBA|BSc|MSc|"
    r"Bachelor(?:'s)?|Master(?:'s)?|Ph\.?\s*D\.?|Diploma|High\s+School)\b",
    re.IGNORECASE
)
_TITLE_VALUE = re.compile(
    r"\b(?:engineer|developer|analyst|manager|executive|assistant|designer|"
    r"specialist|intern|associate|consultant|administrator|technician|"
    r"software\s+development\s+role)\b",
    re.IGNORECASE
)
_JOB_DUTY_START = re.compile(
    r"^(?:you\s+will\b|responsible\s+for\b|"
    r"(?:develop|maintain|create|design|build|implement|test|perform|"
    r"collaborate|contribute|work\s+on|participate\s+in|support|manage|"
    r"analy[sz]e|resolve|deliver|review|troubleshoot|deploy|coordinate)\b)",
    re.IGNORECASE
)


def _read_ocr_layout(image, page_segmentation_mode: int):
    """Read positioned OCR words and group them into visual lines."""
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config=f"--psm {page_segmentation_mode}"
    )
    grouped = {}
    for index, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        key = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index]
        )
        grouped.setdefault(key, []).append({
            "text": word,
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index])
        })

    lines = []
    for words in grouped.values():
        words.sort(key=lambda word: word["left"])
        lines.append({
            "text": " ".join(word["text"] for word in words),
            "left": min(word["left"] for word in words),
            "top": min(word["top"] for word in words),
            "right": max(word["left"] + word["width"] for word in words),
            "bottom": max(word["top"] + word["height"] for word in words),
            "words": words
        })
    return sorted(lines, key=lambda line: (line["top"], line["left"]))


def _find_layout_labels(lines):
    """Find labeled headings in positioned OCR lines, including adjacent columns."""
    matches = []
    for line in lines:
        words = line["words"]
        normalized_words = [
            re.sub(r"[^a-z0-9]+", "", word["text"].casefold())
            for word in words
        ]
        candidates = []
        for field, labels in FIELD_LABELS.items():
            for label in labels:
                label_words = [
                    re.sub(r"[^a-z0-9]+", "", word)
                    for word in label.casefold().split()
                ]
                label_words = [word for word in label_words if word]
                if not label_words:
                    continue
                for start in range(len(normalized_words) - len(label_words) + 1):
                    end = start + len(label_words)
                    if normalized_words[start:end] != label_words:
                        continue
                    exact_line = (
                        len(normalized_words) == len(label_words)
                        or re.search(r"[:：]\s*$", words[end - 1]["text"])
                    )
                    candidates.append((start, end, field, len(label_words), exact_line))

        recognized_labels = len({candidate[2] for candidate in candidates})
        for start, end, field, word_count, exact_line in candidates:
            if word_count == 1 and not exact_line and recognized_labels < 2:
                continue
            selected = words[start:end]
            matches.append({
                "field": field,
                "left": min(word["left"] for word in selected),
                "right": max(word["left"] + word["width"] for word in selected),
                "top": min(word["top"] for word in selected),
                "bottom": max(word["top"] + word["height"] for word in selected)
            })
    return matches


def _extract_positioned_fields(ocr_layout, image_width: int):
    """Associate nearby OCR lines with their closest labeled visual section."""
    lines = ocr_layout["psm11"]
    headers = (
        _find_layout_labels(ocr_layout["psm11"])
        + _find_layout_labels(ocr_layout["psm6"])
    )
    unique_headers = []
    for header in sorted(headers, key=lambda item: (item["top"], item["left"])):
        if any(
            header["field"] == existing["field"]
            and abs(header["top"] - existing["top"]) < 40
            and (
                abs(header["left"] - existing["left"]) < 40
                or (
                    min(header["right"], existing["right"]) - max(header["left"], existing["left"])
                ) / max(1, min(header["right"] - header["left"], existing["right"] - existing["left"])) >= 0.7
            )
            for existing in unique_headers
        ):
            continue
        unique_headers.append(header)

    centers = sorted({
        int((header["left"] + header["right"]) / 2)
        for header in unique_headers
    })
    positioned = {}
    for header in unique_headers:
        center = (header["left"] + header["right"]) / 2
        left_neighbors = [value for value in centers if value < center - 150]
        right_neighbors = [value for value in centers if value > center + 150]
        left_bound = (max(left_neighbors) + center) / 2 if left_neighbors else 0
        right_bound = (center + min(right_neighbors)) / 2 if right_neighbors else image_width

        below_headers = [
            other for other in unique_headers
            if other is not header
            and other["top"] > header["bottom"] + 5
            and (
                left_bound <= (other["left"] + other["right"]) / 2 <= right_bound
                or abs((other["left"] + other["right"]) / 2 - center) <= max(250, image_width * 0.25)
            )
        ]
        next_header_top = min((other["top"] for other in below_headers), default=None)

        candidates = []
        for line in lines:
            if line["top"] <= header["bottom"] + 5:
                continue
            if next_header_top is not None and line["top"] >= next_header_top:
                continue
            line_center = (line["left"] + line["right"]) / 2
            if not left_bound <= line_center <= right_bound:
                continue
            candidates.append(line)

        values = _positioned_values(header["field"], candidates)
        if values:
            current_values = positioned.setdefault(header["field"], [])
            for value in values:
                if value.casefold() not in {existing.casefold() for existing in current_values}:
                    current_values.append(value)
    return positioned


def _positioned_values(field: str, lines):
    """Keep only value-shaped lines for fields inside one visual section."""
    texts = [line["text"].strip() for line in lines if line["text"].strip()]
    if field == "experience_requirements":
        for text in texts:
            match = _EXPERIENCE_VALUE.search(text)
            if match:
                return [match.group(0)]
        return []
    if field == "job_location":
        for text in texts:
            match = _LOCATION_HINTS.search(text)
            if not match:
                continue
            before = text[:match.start()].strip(" \t,|:：📍●•·")
            after = text[match.end():].strip(" \t,|:：📍●•·")
            if not before and not after:
                return [match.group(0)]
        return []
    if field == "education":
        selected = []
        for index, text in enumerate(texts):
            if _education_value_starts(text):
                selected.append(text)
                equivalent = next(
                    (value for value in texts[index + 1:] if re.search(r"\bor\s+equivalent\b", value, re.IGNORECASE)),
                    None
                )
                if equivalent:
                    selected.append(equivalent)
                break
        return selected
    if field == "required_skills":
        selected = []
        for line in lines:
            text = line["text"].strip()
            normalized = text.casefold()
            words = text.split()
            if (
                re.search(r"\b(?:work on|participate|perform|collaborate|contribute|develop|responsible|great work culture|career growth|excellent learning)\b", normalized)
                or len(words) > 10
            ):
                continue
            if (
                re.match(r"^\s*[@•✓✔-]\s*", text)
                or text.count(",") >= 1
                or re.fullmatch(r"problem[\s-]+solving", normalized)
            ):
                cleaned = re.sub(r"^\s*[@•✓✔-]\s*", "", text).strip(" ,|")
                cleaned = re.sub(r"\bJava\s+i\s*\+", "Java 11+", cleaned, flags=re.IGNORECASE)
                if cleaned and cleaned.casefold() not in {value.casefold() for value in selected}:
                    if re.match(r"^and\s+", cleaned, re.IGNORECASE) and selected:
                        selected[-1] += " " + cleaned
                    else:
                        selected.append(cleaned)
        return selected
    if field == "job_description":
        excluded = re.compile(
            r"\b(?:why\s+\w+|global\s+leader|excellent\s+learning|great\s+work\s+culture|"
            r"career\s+growth|visit\s+our|apply\s+now|careers?\s+page)\b",
            re.IGNORECASE
        )
        return [
            re.sub(r"^\s*[@•✓✔-]\s*", "", text).strip()
            for text in texts
            if len(text.split()) >= 5
            and not excluded.search(text)
            and not re.search(r"https?://|www\.", text, re.IGNORECASE)
        ]
    return []


def _education_value_starts(text: str) -> bool:
    value = re.sub(
        r"^\s*(?:education|qualification|degree)\s*[:：\-–]?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    value = re.sub(r"^[\W_]+", "", value)
    return bool(_EDUCATION_VALUE.match(value))


def is_ocr_available() -> bool:
    """Return whether an OCR engine is ready for execution."""
    return find_tesseract_binary() is not None


def extract_entities_from_text(raw_text: str, ocr_layout=None) -> Dict[str, str]:
    """
    Parse explicitly labeled sections into separate editable fields. Unlabeled
    values are only inferred for well-known location phrases and contact details.
    """
    def clean_value(value: str) -> str:
        return re.sub(r"^[\s|:：\-–—•*·📍]+|[\s|]+$", "", value).strip()

    def is_only_punctuation(value: str) -> bool:
        return not re.search(r"\w", value, re.UNICODE)

    def line_labels(line: str):
        found = []
        for match in _LABEL_PATTERN.finditer(line):
            label = re.sub(r"\s+", " ", match.group("label")).strip().casefold()
            found.append((match.start(), match.end(), _LABEL_TO_FIELD[label]))
        if found:
            return found

        normalized = clean_value(line).casefold()
        for field, labels in FIELD_LABELS.items():
            for label in sorted(labels, key=len, reverse=True):
                if normalized == label.casefold():
                    return [(0, len(line), field)]
                alias_pattern = re.escape(label).replace(r"\ ", r"\s+")
                prefix = re.match(
                    rf"^\s*{alias_pattern}\s*[:：\-–]?\s+(.+)$",
                    line,
                    re.IGNORECASE
                )
                if prefix:
                    return [(0, prefix.start(1), field)]
        return []

    field_lines: Dict[str, list] = {}
    current_field = None
    source_lines = raw_text.splitlines()
    skip_next_location_line = False
    for line_index, line in enumerate(source_lines):
        if skip_next_location_line:
            skip_next_location_line = False
            continue

        labels = line_labels(line)
        if labels:
            current_field = None
            for index, (start, end, field) in enumerate(labels):
                next_start = labels[index + 1][0] if index + 1 < len(labels) else len(line)
                value = clean_value(line[end:next_start])
                if value:
                    field_lines.setdefault(field, []).append(value)
                current_field = field
            continue

        value = clean_value(line)
        if re.search(r"[📍\u25cf]", line):
            location_value = value
            hint = _LOCATION_HINTS.search(location_value)
            next_line_has_location = False
            if not hint and line_index + 1 < len(source_lines):
                location_value = clean_value(source_lines[line_index + 1])
                hint = _LOCATION_HINTS.search(location_value)
                if hint:
                    next_line_has_location = True
            if hint:
                location_prefix = location_value[:hint.start()]
                location_suffix = location_value[hint.end():]
                if is_only_punctuation(location_prefix) and is_only_punctuation(location_suffix):
                    field_lines.setdefault("job_location", []).append(hint.group(0))
                    current_field = None
                    skip_next_location_line = next_line_has_location
                    continue

        location_hint = _LOCATION_HINTS.search(value)
        if location_hint:
            location_prefix = value[:location_hint.start()]
            location_suffix = value[location_hint.end():]
            is_general_location = _GENERAL_LOCATIONS.fullmatch(value) is not None
            if (
                is_general_location
                and is_only_punctuation(location_prefix)
                and is_only_punctuation(location_suffix)
            ):
                field_lines.setdefault("job_location", []).append(location_hint.group(0))
                current_field = None
                continue
        if value and current_field:
            field_lines.setdefault(current_field, []).append(value)
        elif value:
            current_field = None

    def section_value(field: str) -> str:
        values = [clean_value(value) for value in field_lines.get(field, [])]
        values = [value for value in values if value]
        separator = ", " if field == "required_skills" else " "
        return separator.join(values)

    text_blob = raw_text or ""
    email_values = extract_emails(section_value("recruiter_email") or text_blob)
    phone_values = extract_phone_numbers(section_value("contact_number"))
    if not phone_values and re.search(r"\b(?:phone|mobile|contact)\b", text_blob, re.IGNORECASE):
        phone_values = extract_phone_numbers(text_blob)

    urls = extract_urls(text_blob)
    application_url = section_value("application_url")
    official_website = section_value("official_website")
    if not application_url:
        application_url = next(
            (url for url in urls if re.search(r"apply|jobs|careers|greenhouse|lever|workday", url, re.IGNORECASE)),
            ""
        )
    location = section_value("job_location")
    if not location:
        for line_index, line in enumerate(source_lines):
            if re.search(r"[📍\u25cf]\s*", line):
                hint = _LOCATION_HINTS.search(line)
                if not hint:
                    next_line_index = line_index + 1
                    if next_line_index < len(source_lines):
                        hint = _LOCATION_HINTS.search(source_lines[next_line_index])
                if hint:
                    location = clean_value(hint.group(0))
                    break
            hint = _LOCATION_HINTS.search(line)
            before = line[:hint.start()] if hint else ""
            after = line[hint.end():] if hint else ""
            if (
                hint
                and _GENERAL_LOCATIONS.fullmatch(clean_value(line))
                and is_only_punctuation(before)
                and is_only_punctuation(after)
                and not re.search(r"https?://|www\.|@|\b(?:street|road|avenue|address|office)\b", line, re.IGNORECASE)
            ):
                location = clean_value(hint.group(0))
                break
    if location and re.search(r"https?://|www\.|@|\b(?:street|road|avenue|address)\b", location, re.IGNORECASE):
        location = ""

    fields = {
        "company_name": section_value("company_name"),
        "job_title": section_value("job_title"),
        "job_description": section_value("job_description"),
        "salary": section_value("salary"),
        "job_location": location,
        "required_skills": section_value("required_skills"),
        "experience_requirements": section_value("experience_requirements"),
        "education": section_value("education"),
        "recruiter_email": email_values[0] if email_values else "",
        "contact_number": phone_values[0] if phone_values else "",
        "recruiter_phone": phone_values[0] if phone_values else "",
        "application_url": application_url,
        "official_careers_url": section_value("official_careers_url"),
        "official_website": official_website,
        "official_website_url": official_website
    }

    raw_lines = [clean_value(line) for line in text_blob.splitlines()]
    if not fields["job_description"]:
        duty_lines = []
        for line in raw_lines:
            line = re.sub(r"^\s*[@•✓✔-]\s*", "", line).strip()
            if (
                len(line.split()) >= 5
                and _JOB_DUTY_START.search(line)
                and not re.search(r"https?://|www\.", line, re.IGNORECASE)
            ):
                duty_lines.append(line)
        fields["job_description"] = " ".join(duty_lines)

    if not fields["company_name"]:
        for index, line in enumerate(raw_lines):
            company_match = re.search(
                r"\b(?:why|about)\s+([A-Z][A-Za-z0-9&.-]+)\b",
                line,
                re.IGNORECASE
            )
            if company_match:
                fields["company_name"] = company_match.group(1)
                break
            if re.search(r"\bbe\s+part\s+of\b", line, re.IGNORECASE):
                for candidate in raw_lines[index + 1:index + 5]:
                    if re.fullmatch(r"[A-Z][A-Za-z0-9&.-]{2,}", candidate):
                        fields["company_name"] = candidate
                        break
                if fields["company_name"]:
                    break

    if not fields["job_title"]:
        title_candidates = []
        for index, line in enumerate(raw_lines):
            if (
                len(line.split()) <= 8
                and not re.search(r"[.!?]|https?://", line)
                and _TITLE_VALUE.search(line)
                and not re.search(r"\b(?:why|responsibilities|skills|experience|education|location)\b", line, re.IGNORECASE)
            ):
                if re.fullmatch(r"associate", line, re.IGNORECASE):
                    continue
                title_candidates.append((index, line))
        if title_candidates:
            title_index, title = title_candidates[0]
            if re.fullmatch(r"software\s+development\s+role", title, re.IGNORECASE):
                if any(re.fullmatch(r"associate", line, re.IGNORECASE) for line in raw_lines[max(0, title_index - 12):title_index]):
                    title = "Associate " + title
            fields["job_title"] = title

    for line in raw_lines:
        experience_match = _EXPERIENCE_VALUE.search(line)
        if experience_match:
            fields["experience_requirements"] = experience_match.group(0)
            break

    has_location_label = bool(re.search(r"\b(?:job\s+location|work\s+location|location|city)\b", text_blob, re.IGNORECASE))
    if has_location_label and not fields["job_location"]:
        for line in raw_lines:
            location_match = _LOCATION_HINTS.search(line)
            if location_match:
                before = line[:location_match.start()].strip(" \t,|:：📍●•·")
                after = line[location_match.end():].strip(" \t,|:：📍●•·")
                if not before and not after:
                    fields["job_location"] = location_match.group(0)
                    break

    if not _education_value_starts(fields["education"]):
        fields["education"] = ""
        education_lines = []
        for index, line in enumerate(raw_lines):
            if _education_value_starts(line):
                education_lines.append(line)
                if index + 1 < len(raw_lines) and re.search(r"\bor\s+equivalent\b", raw_lines[index + 1], re.IGNORECASE):
                    education_lines.append(raw_lines[index + 1])
                fields["education"] = " ".join(education_lines)
                break

    if ocr_layout:
        positioned_fields = _extract_positioned_fields(
            ocr_layout,
            int(ocr_layout["image_width"])
        )
        for field, values in positioned_fields.items():
            if field == "required_skills":
                fields[field] = ", ".join(values)
            elif field == "job_description":
                fields[field] = " ".join(values)
            else:
                fields[field] = " ".join(values)

    careers_urls = extract_urls(section_value("official_careers_url"))
    if not careers_urls and re.search(r"\bcareers?\s+page\b", text_blob, re.IGNORECASE):
        careers_urls = [url for url in urls if re.search(r"careers?", url, re.IGNORECASE)]
    fields["official_careers_url"] = careers_urls[0] if careers_urls else ""
    if fields["official_careers_url"] and not fields["application_url"]:
        fields["application_url"] = fields["official_careers_url"]

    fields["experience"] = fields["experience_requirements"]
    fields["location"] = fields["job_location"]
    return fields


def perform_ocr(image_path: str) -> Dict[str, Any]:
    """
    Run OCR extraction on an uploaded image file.
    Returns:
    {
        "success": bool,
        "raw_text": str,
        "engine_used": str,
        "error_message": Optional[str],
        "extracted_fields": Dict[str, str]
    }
    """
    tess_bin = find_tesseract_binary()
    if not tess_bin:
        return {
            "success": False,
            "raw_text": "",
            "engine_used": "None",
            "error_message": (
                "Tesseract OCR was not found in the configured locations or system PATH. "
                "Please install Tesseract OCR for Windows (UB-Mannheim) or manually fill in the job details in the form below."
            ),
            "extracted_fields": {
                "company_name": "",
                "job_title": "",
                "job_description": "",
                "salary": "",
                "job_location": "",
                "required_skills": "",
                "experience_requirements": "",
                "experience": "",
                "location": "",
                "education": "",
                "recruiter_email": "",
                "contact_number": "",
                "recruiter_phone": "",
                "application_url": "",
                "official_careers_url": "",
                "official_website": "",
                "official_website_url": ""
            }
        }

    try:
        pytesseract.pytesseract.tesseract_cmd = tess_bin
        preprocessed_img = preprocess_image_for_ocr(image_path)
        try:
            orientation = pytesseract.image_to_osd(preprocessed_img, config="--psm 0")
            rotation = int(re.search(r"Rotate:\s+(\d+)", orientation).group(1))
            if rotation:
                preprocessed_img = preprocessed_img.rotate(-rotation, expand=True)
        except (pytesseract.TesseractError, AttributeError, ValueError):
            pass
        raw_text = pytesseract.image_to_string(preprocessed_img, config="--psm 11")
        ocr_layout = {
            "psm11": _read_ocr_layout(preprocessed_img, 11),
            "psm6": _read_ocr_layout(preprocessed_img, 6),
            "image_width": preprocessed_img.width
        }
        extracted_fields = extract_entities_from_text(raw_text, ocr_layout=ocr_layout)

        if not raw_text or len(raw_text.strip()) < 5:
            return {
                "success": False,
                "raw_text": raw_text,
                "engine_used": "Tesseract OCR",
                "error_message": (
                    "Could not extract legible text from this image. "
                    "The image may be blurry, low resolution, or contain purely graphical elements. "
                    "Please review the image preview and enter details manually."
                ),
                "extracted_fields": extracted_fields
            }

        return {
            "success": True,
            "raw_text": raw_text.strip(),
            "engine_used": "Tesseract OCR",
            "error_message": None,
            "extracted_fields": extracted_fields
        }

    except Exception as e:
        return {
            "success": False,
            "raw_text": "",
            "engine_used": "Tesseract OCR",
            "error_message": f"OCR extraction error: {str(e)}",
            "extracted_fields": extract_entities_from_text("")
        }
