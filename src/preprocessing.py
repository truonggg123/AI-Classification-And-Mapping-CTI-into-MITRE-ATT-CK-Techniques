"""
CTI & MITRE ATT&CK Preprocessing Module
Chứa các hàm làm sạch, trung hòa thực thể và chuẩn hóa dữ liệu CTI theo domain knowledge mới nhất.
Được đồng bộ với logic từ notebook 02_preprocessing.ipynb.

Output chuẩn gồm:
- Cleaned_Text: text đã chuẩn hóa, phù hợp cho Transformer/embedding model.
- Tokenized_Text: chuỗi token CTI thống nhất, phù hợp cho vectorizer/classical model.
- Labels: MITRE ATT&CK parent techniques.
"""

import argparse
import json
import re
import html
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_STAGE1_THRESHOLD = 30
CTI_TOKEN_PATTERN = r"[a-z0-9_]+(?:[./:-][a-z0-9_]+)*"

# Cấu hình ánh xạ giảm nhãn cho các mẫu có số lượng nhãn [4, 5, 6, 8, 15] từ notebook 01
RARE_LABEL_MAPPING = {
    "T1021,T1046,T1078,T1210": "T1021,T1046,T1078",
    "T1021,T1040,T1210,T1563": "T1021,T1040,T1563",
    "T1043,T1059,T1071,T1571": "T1059,T1071,T1571",
    "T1048,T1059,T1071,T1090": "T1059,T1071,T1090",
    "T1020,T1048,T1095,T1572": "T1048,T1095,T1572",
    "T1053,T1055,T1547,T1564": "T1053,T1547,T1564",
    "T1021,T1053,T1059,T1105": "T1021,T1059,T1105",
    "T1021,T1059,T1075,T1105": "T1021,T1059,T1105",
    "T1018,T1033,T1057,T1082,T1087": "T1018,T1082,T1087",
    "T1037,T1053,T1204,T1546": "T1037,T1053,T1546",
    "T1202,T1218,T1546,T1556": "T1202,T1546,T1556",
    "T1056,T1071,T1219,T1546": "T1056,T1071,T1219",
    "T1056,T1204,T1546,T1556": "T1056,T1546,T1556",
    "T1078,T1086,T1552,T1557": "T1078,T1086,T1552",
    "T1021,T1071,T1552,T1566": "T1021,T1071,T1552",
    "T1059,T1190,T1203,T1552": "T1059,T1190,T1552",
    "T1078,T1210,T1486,T1530": "T1078,T1486,T1530",
    "T1033,T1041,T1053,T1082": "T1033,T1041,T1082",
    "T1016,T1078,T1210,T1570": "T1078,T1210,T1570",
    "T1059,T1218,T1543,T1569,T1570": "T1543,T1569,T1570",
    "T1041,T1056,T1105,T1113": "T1041,T1056,T1113",
    "T1056,T1078,T1082,T1566": "T1056,T1078,T1082",
    "T1027,T1059,T1078,T1106,T1140,T1548": "T1078,T1140,T1548",
    "T1005,T1057,T1083,T1518": "T1057,T1083,T1518",
    "T1016,T1033,T1057,T1071,T1082,T1573": "T1016,T1033,T1082",
    "T1027,T1070,T1112,T1140": "T1070,T1112,T1140",
    "T1059,T1106,T1204,T1547": "T1059,T1106,T1204",
    "T1053,T1140,T1543,T1574": "T1053,T1543,T1574",
    "T1016,T1027,T1033,T1055,T1071,T1082": "T1027,T1055,T1071",
    "T1003,T1027,T1041,T1047,T1053,T1078,T1112,T1543": "T1003,T1041,T1543",
    "T1041,T1059,T1071,T1105,T1113,T1547": "T1041,T1059,T1113",
    "T1027,T1105,T1204,T1218": "T1105,T1204,T1218",
    "T1016,T1057,T1082,T1083": "T1016,T1057,T1082",
    "T1041,T1057,T1083,T1105,T1113": "T1041,T1057,T1113",
    "T1059,T1071,T1105,T1140": "T1059,T1105,T1140",
    "T1027,T1047,T1055,T1059,T1105,T1218": "T1047,T1055,T1059",
    "T1016,T1033,T1047,T1057,T1082": "T1016,T1057,T1082",
    "T1021,T1055,T1078,T1570": "T1021,T1078,T1570",
    "T1055,T1057,T1059,T1070,T1083,T1105": "T1055,T1059,T1105",
    "T1016,T1059,T1082,T1547": "T1016,T1082,T1547",
    "T1059,T1105,T1106,T1140": "T1059,T1105,T1140",
    "T1016,T1033,T1082,T1518,T1547": "T1016,T1082,T1547",
    "T1057,T1059,T1070,T1083,T1105,T1113": "T1057,T1083,T1113",
    "T1016,T1033,T1057,T1082,T1573": "T1016,T1033,T1082",
    "T1012,T1016,T1057,T1082,T1113": "T1012,T1057,T1113",
    "T1078,T1543,T1569,T1570": "T1078,T1543,T1570",
    "T1003,T1005,T1016,T1047,T1053,T1059,T1068,T1071,T1078,T1082,T1140,T1210,T1547,T1570,T1573": "T1053,T1059,T1078",
    "T1047,T1055,T1082,T1218": "T1047,T1055,T1218",
    "T1027,T1053,T1055,T1059,T1112": "T1027,T1053,T1055",
    "T1056,T1059,T1112,T1113": "T1056,T1059,T1113",
    "T1012,T1055,T1056,T1071,T1090,T1112,T1113,T1548": "T1055,T1071,T1090",
    "T1005,T1012,T1057,T1070,T1112": "T1005,T1057,T1112",
    "T1005,T1041,T1056,T1071": "T1041,T1056,T1071",
    "T1057,T1068,T1190,T1518,T1566": "T1057,T1190,T1566",
}


def cti_tokenizer(text):
    """
    Tokenizer CTI dùng chung cho toàn nhóm.

    Giữ từ, số, underscore và chuỗi kỹ thuật có dấu `.`, `/`, `:`, `-`
    như cmd.exe, URL_TOKEN, CVE_YEAR_2021 hoặc /etc/passwd.
    """
    return re.findall(CTI_TOKEN_PATTERN, str(text).lower())


def tokenize_cti_text(text):
    """Chuyển Cleaned_Text thành chuỗi token cách nhau bằng một khoảng trắng."""
    return " ".join(cti_tokenizer(text))


def remove_procedural_noise(text):
    """
    Xóa nhiễu dạng hướng dẫn từng bước, tiêu đề template, cụm lặp lại.
    Không xóa chữ 'step' khi nó nằm trong câu tự nhiên.
    """
    text = str(text)

    # Tách heading Step/Phase bị dính vào từ trước: "logsStep 3" -> "logs. Step 3"
    text = re.sub(
        r"(?<=[a-z)])(?=(?:Step|Phase)\s+\d+\b)",
        ". ",
        text
    )

    # Xóa numbering bị dính sau dấu chấm: "collector.2. Load" -> "collector. Load"
    text = re.sub(
        r"(?<=[A-Za-z)])\.\s*\d{1,2}\.\s*",
        ". ",
        text
    )

    # Xóa số thứ tự dạng "1. Find", "10. Cleanup"
    # Không ăn nhầm IP vì sau dấu chấm phải có khoảng trắng
    text = re.sub(
        r"(?m)(^|\s)\d{1,2}\.\s+",
        " ",
        text
    )

    # Xóa số thứ tự bị dính sau dấu câu: "database.2. Send" -> "database. Send"
    # Không ăn nhầm IP vì trước số phải là chữ hoặc dấu đóng ngoặc, không phải số
    text = re.sub(
        r"(?<=[A-Za-z\)])\d{1,2}\.\s*",
        " ",
        text
    )

    # Chỉ xóa Step/Phase khi có số: Step 1, STEP 10, Phase 2
    text = re.sub(
        r"\b(step|phase)\s+\d+\s*[:\-–—]?",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Một số heading template bị dính với động từ kế tiếp: "CleanupRemove"
    text = re.sub(r"Cleanup(?=[A-Z])", " ", text)

    boilerplate_phrases = [
        r"\bAI Agents\s*&\s*LLM Exploits\b",
        r"\bAI/ML Security\b",
        r"\bRed Team\b",
        r"\bBlue Team\b",
        r"\bPCAP Dataset\b",
        r"\bBeginner Friendly\b",
        r"\bOptional\b",
        r"\bCleanup\b",
        r"\bSetup Lab\b",
        r"\bLab Setup\b",
        r"\bTools like\b",
        r"\bYou now have\b",
        r"\bThis simulation demonstrates how\b",
        r"\bThis simulation shows how\b",
        r"\bThis simulation demonstrates\b",
        r"\bThis simulation shows\b",
        r"\bGreat for demos\b",
        r"\bGreat for dataset generation\b",
    ]

    for phrase in boilerplate_phrases:
        text = re.sub(
            phrase + r"\s*[.:]?",
            " ",
            text,
            flags=re.IGNORECASE
        )

    # Chuẩn hóa một số cụm bị tách sai
    text = re.sub(r"\bwi\s*fi\b", "wifi", text, flags=re.IGNORECASE)
    text = re.sub(r"\bci\s*cd\b", "cicd", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpost\s+quantum\b", "postquantum", text, flags=re.IGNORECASE)

    # Cleanup dấu câu rỗng
    text = re.sub(r"\s+\.\s+", ". ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_mitre_id_leakage_from_text(text):
    """
    Trung hòa MITRE ATT&CK Technique IDs trong input text để tránh data leakage.
    Không áp dụng cho cột Labels.
    """
    text = str(text)

    # Trung hòa MITRE technique URLs dạng:
    # https://attack.mitre.org/techniques/T1059/
    # https://attack.mitre.org/techniques/T1059/001/
    text = re.sub(
        r"https?://attack\.mitre\.org/techniques/T\d{4}(?:/\d{3})?/?",
        " MITRE_TECHNIQUE_REF ",
        text,
        flags=re.IGNORECASE
    )

    # Trung hòa mọi Technique ID dạng Txxxx hoặc Txxxx.xxx
    text = re.sub(
        r"\bT\d{4}(?:\.\d{3})?\b",
        " MITRE_TECHNIQUE_ID ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_sqli_signal_tokens(text):
    """
    Thêm các token tín hiệu SQL Injection trước khi làm sạch ký tự đặc biệt.
    Không xóa payload gốc ngay, chỉ append token vào cuối text.
    """
    tokens = []
    text_str = str(text)
    lower_text = text_str.lower()

    has_sqli_context = bool(
        re.search(
            r"(?:\b(?:sql injection|sqli|blind sql|time[- ]based sql|sqlmap|sqlninja|"
            r"union\s+(?:all\s+)?select|information_schema|waitfor\s+delay|pg_sleep)\b|"
            r"@@version|\bselect\b.{0,120}\bfrom\b)",
            lower_text,
            flags=re.IGNORECASE
        )
    )

    # Boolean-based SQLi
    if re.search(r"\b(or|and)\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?(?!\w)", text_str, flags=re.IGNORECASE):
        tokens.append("SQLI_BOOLEAN_TRUE")

    if re.search(r"\b(or|and)\s+['\"]?1['\"]?\s*=\s*['\"]?2['\"]?(?!\w)", text_str, flags=re.IGNORECASE):
        tokens.append("SQLI_BOOLEAN_FALSE")

    # SQL comment markers: --, /*, */
    if has_sqli_context and re.search(r"(--|/\*|\*/)", text_str):
        tokens.append("SQLI_COMMENT")

    # Time-based SQLi
    if has_sqli_context and re.search(
        r"\b(sleep|benchmark|pg_sleep|waitfor\s+delay)\s*\(",
        text_str,
        flags=re.IGNORECASE
    ):
        tokens.append("SQLI_TIME_DELAY")

    # UNION-based SQLi
    if re.search(r"\bunion\s+(all\s+)?select\b", text_str, flags=re.IGNORECASE):
        tokens.append("SQLI_UNION_SELECT")

    # SQL enumeration functions
    if has_sqli_context and re.search(
        r"\b(substring|substr|ascii|length|database|version|user|schema_name)\s*\(",
        text_str,
        flags=re.IGNORECASE
    ):
        tokens.append("SQLI_ENUMERATION_FUNC")

    # Database version variable
    if re.search(r"@@version", text_str, flags=re.IGNORECASE):
        tokens.append("SQLI_DB_VERSION")

    # Generic SQLi context
    if has_sqli_context:
        tokens.append("SQLI_CONTEXT")

    if tokens:
        text_str = text_str + " " + " ".join(sorted(set(tokens)))

    return text_str


def normalize_sqli_payloads(text):
    text = str(text)
    has_sqli_context = "SQLI_CONTEXT" in text

    # Conditional time-based SQLi: IF(1=1, SLEEP(5), 0)
    text = re.sub(
        r"\bif\s*\([^)]*(sleep|benchmark|pg_sleep)\s*\([^)]*\)[^)]*\)",
        " SQLI_CONDITIONAL_TIME_DELAY " if has_sqli_context else r"\g<0>",
        text,
        flags=re.IGNORECASE
    )

    # Boolean-based SQLi
    text = re.sub(
        r"\b(or|and)\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?(?!\w)",
        " SQLI_BOOLEAN_TRUE ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b(or|and)\s+['\"]?1['\"]?\s*=\s*['\"]?2['\"]?(?!\w)",
        " SQLI_BOOLEAN_FALSE ",
        text,
        flags=re.IGNORECASE
    )

    # Time-based SQLi
    text = re.sub(
        r"\b(sleep|benchmark|pg_sleep)\s*\([^)]*\)",
        " SQLI_TIME_DELAY " if has_sqli_context else r"\g<0>",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bwaitfor\s+delay\b",
        " SQLI_TIME_DELAY " if has_sqli_context else r"\g<0>",
        text,
        flags=re.IGNORECASE
    )

    # UNION SELECT
    text = re.sub(
        r"\bunion\s+(all\s+)?select\b",
        " SQLI_UNION_SELECT ",
        text,
        flags=re.IGNORECASE
    )

    # Enumeration functions
    text = re.sub(
        r"\b(substring|substr|ascii|length|database|version|user|schema_name)\s*\([^)]*\)",
        " SQLI_ENUMERATION_FUNC " if has_sqli_context else r"\g<0>",
        text,
        flags=re.IGNORECASE
    )

    # Database version variable
    text = re.sub(
        r"@@version",
        " SQLI_DB_VERSION ",
        text,
        flags=re.IGNORECASE
    )

    # SQL comment markers
    text = re.sub(
        r"(--|/\*|\*/)",
        " SQLI_COMMENT " if has_sqli_context else r"\g<0>",
        text
    )

    # Mỗi signal chỉ cần xuất hiện một lần; tránh token vừa replace vừa append bị lặp.
    seen_signals = set()

    def keep_first_signal(match):
        token = match.group(0)
        if token in seen_signals:
            return " "
        seen_signals.add(token)
        return token

    text = re.sub(r"\bSQLI_[A-Z_]+\b", keep_first_signal, text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_cve(text):
    """
    CVE-2021-44228 -> CVE_TOKEN CVE_YEAR_2021
    Giữ tín hiệu có CVE và giữ năm, không để model học vẹt mã CVE cụ thể.
    """
    def repl(match):
        cve = match.group(0).upper()
        year = cve.split("-")[1]
        return f" CVE_TOKEN CVE_YEAR_{year} "

    return re.sub(
        r"\bCVE-\d{4}-\d{4,7}\b",
        repl,
        str(text),
        flags=re.IGNORECASE
    )


def normalize_windows_path(text):
    """
    Chuẩn hóa Windows path nhưng giữ tín hiệu quan trọng.
    Bản này xử lý được path có dấu cách như Start Menu, Program Files.
    """
    text = str(text)

    # Bảo vệ các folder Windows có dấu cách để regex không bị cắt giữa chừng
    protected_phrases = {
        "Start Menu": "Start_Menu",
        "Program Files (x86)": "Program_Files_x86",
        "Program Files": "Program_Files",
        "Common Files": "Common_Files",
    }

    for original, protected in protected_phrases.items():
        text = re.sub(
            re.escape(original),
            protected,
            text,
            flags=re.IGNORECASE
        )

    def repl(match):
        path = match.group(0)
        path_lower = path.lower()

        tokens = ["WINDOWS_PATH"]

        if "system32" in path_lower:
            tokens.append("system32")

        if "syswow64" in path_lower:
            tokens.append("syswow64")

        if "startup" in path_lower:
            tokens.append("startup_folder")

        if "appdata" in path_lower:
            tokens.append("appdata")

        if "temp" in path_lower:
            tokens.append("temp_dir")

        if "programdata" in path_lower:
            tokens.append("programdata")

        if "program_files" in path_lower:
            tokens.append("program_files")

        # Lấy basename cuối path
        basename = re.split(r"[\\/]", path)[-1]
        basename = basename.strip().lower()

        if basename and "." in basename:
            tokens.append(basename)

        return " " + " ".join(dict.fromkeys(tokens)) + " "

    # Chuẩn hóa UNC path (\\server\share\file) nhưng giữ basename có extension
    def repl_unc(match):
        path = match.group(0)
        basename = re.split(r"[\\/]", path)[-1].strip().lower()
        tokens = ["UNC_PATH", "network_share"]
        if basename and "." in basename:
            tokens.append(basename)
        return " " + " ".join(dict.fromkeys(tokens)) + " "

    text = re.sub(
        r"(?<!\\)\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9$_.-]+(?:\\[^\s\"']*)?",
        repl_unc,
        text
    )

    # Chuẩn hóa drive root đứng riêng như C:\ hoặc Z:\
    text = re.sub(
        r"(?<![A-Za-z0-9])[A-Za-z]:\\(?=\s|$|[;,)])",
        " WINDOWS_PATH drive_root ",
        text
    )

    # Bắt Windows path sau khi đã bảo vệ dấu cách
    text = re.sub(
        r"[A-Za-z]:\\[^\s\"']+",
        repl,
        text
    )

    return text


def normalize_unix_path(text):
    """
    Chuẩn hóa Unix/Linux path thật, tránh ăn nhầm Windows command options như:
    /c, /all, /node, /quiet, /S, /Q.
    """
    text = str(text)

    important_paths = {
        "/root/.ssh/authorized_keys": "UNIX_PATH ssh_authorized_keys",
        "/etc/passwd": "UNIX_PATH etc_passwd",
        "/etc/shadow": "UNIX_PATH etc_shadow",
        "/etc/cron": "UNIX_PATH cron_path",
        "/var/log": "UNIX_PATH log_dir",
        "/tmp": "UNIX_PATH tmp_dir",
        "/home": "UNIX_PATH home_dir",
    }

    for path, token in sorted(important_paths.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(path, f" {token} ")

    unix_root_dirs = r"(etc|var|tmp|root|home|usr|bin|sbin|opt|dev|proc|sys|lib|lib64|mnt|media|srv)"

    text = re.sub(
        rf"(?<!\w)/{unix_root_dirs}(?:/[A-Za-z0-9._-]+)*",
        " UNIX_PATH ",
        text
    )

    return text


def normalize_registry_key(text):
    """
    Chuẩn hóa Registry nhưng giữ tín hiệu Run key, Services key nếu có.
    """
    def repl(match):
        key = match.group(0)
        key_lower = key.lower()

        tokens = ["REGISTRY_KEY"]

        if "\\run" in key_lower or "\\runonce" in key_lower:
            tokens.append("run_key")

        if "currentversion" in key_lower:
            tokens.append("currentversion")

        if "services" in key_lower:
            tokens.append("services_key")

        return " " + " ".join(dict.fromkeys(tokens)) + " "

    return re.sub(
        r"\b(HKLM|HKCU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER)\\[^\s\"']+",
        repl,
        str(text),
        flags=re.IGNORECASE
    )


def normalize_cti_text(text):
    """
    Hàm chuẩn hóa tổng thể một câu/đoạn văn bản CTI theo pipeline 11 bước.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Decode HTML entities nếu có
    text = html.unescape(text)

    # 2. Chuẩn hóa unicode
    text = unicodedata.normalize("NFKC", text)

    # 3. Tránh bị leak nhãn trong cột text
    text = remove_mitre_id_leakage_from_text(text)

    # 4. Xóa nhiễu procedural/template trước
    text = remove_procedural_noise(text)

    # 5. Thêm token tín hiệu SQLi
    text = add_sqli_signal_tokens(text)

    # 6. Chuẩn hóa payload SQLi thành token
    text = normalize_sqli_payloads(text)

    # 7. Chuẩn hóa CVE nhưng giữ năm
    text = normalize_cve(text)

    # 8. Chuẩn hóa URL, email, IP, hash
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL_TOKEN ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL_TOKEN ",
        text
    )

    text = re.sub(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        " IPV4_TOKEN ",
        text
    )

    text = re.sub(
        r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b",
        " HASH_TOKEN ",
        text
    )

    # 9. Registry trước Windows path để tránh regex path ăn mất
    text = normalize_registry_key(text)

    # Chuẩn hóa path
    text = normalize_windows_path(text)
    text = normalize_unix_path(text)

    # 10. Xóa emoji / ký hiệu trang trí phổ biến
    text = re.sub(r"[✅🟢🔺⚠️👉➡️🧠🛑📌•]+", " ", text)

    # Xóa markdown heading / separator
    text = re.sub(r"#{1,6}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"_{3,}", " ", text)

    # Xóa nhãn bước còn sót lại
    text = re.sub(
        r"\b(step|phase)\s*\d+\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Xóa quote/backtick/smart quotes sau khi đã giữ SQLi signals
    text = text.replace("`", " ")
    text = text.replace('"', " ")
    text = text.replace("'", " ")
    text = text.replace("“", " ")
    text = text.replace("”", " ")
    text = text.replace("‘", " ")
    text = text.replace("’", " ")

    # 11. Chạy lại procedural cleanup vì việc bỏ quote/thay token có thể làm lộ numbering bị dính.
    text = remove_procedural_noise(text)

    # Cleanup dấu câu rỗng do xóa template để lại
    text = re.sub(r"\s+\.\s+", ". ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)

    # Xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text).strip()

    # Xóa dấu câu ở đầu/cuối nếu bị dư
    text = text.strip(" .,-;:")

    return text


def normalize_labels(label_str):
    """
    Chuẩn hóa nhãn ATT&CK IDs: gom nhãn con về nhãn cha (Txxxx.xxx -> Txxxx)
    và loại bỏ nhãn trùng lặp.
    """
    if pd.isna(label_str):
        return ""

    label_str = str(label_str)

    # Tách bằng dấu phẩy hoặc chấm phẩy
    parts = re.split(r"[,;]", label_str)

    cleaned = []

    for label in parts:
        label = label.strip().upper()

        # Nhận Txxxx hoặc Txxxx.xxx, sau đó gom về parent Txxxx
        match = re.match(r"^(T\d{4})(?:\.\d{3})?$", label)

        if match:
            parent = match.group(1)
            cleaned.append(parent)

    # Xóa trùng nhưng giữ thứ tự xuất hiện
    cleaned = list(dict.fromkeys(cleaned))

    return ",".join(cleaned)


def label_set(label_str):
    """Chuyển chuỗi nhãn phân tách bằng dấu phẩy thành tập nhãn."""
    return {
        label.strip()
        for label in str(label_str).split(",")
        if label.strip()
    }


def merge_label_strings(label_series):
    """Union các phiên bản nhãn của cùng một Cleaned_Text và giữ thứ tự."""
    merged = []
    for label_str in label_series:
        for label in str(label_str).split(","):
            label = label.strip()
            if label and label not in merged:
                merged.append(label)
    return ",".join(merged)


def manual_resolve_hard_conflict(text, merged_labels):
    """Các rule conflict đã được review thủ công trong notebook 02."""
    lowered = str(text).lower()

    if "downloading and executing two other powershell" in lowered:
        return "T1059,T1105"
    if "passes the obfuscated string to a function" in lowered:
        return "T1027,T1140"
    if "steal credentials by decrypting them from regi" in lowered:
        return "T1005"
    if "used https reverse proxies to redirect c2 traffic" in lowered:
        return "T1071,T1090"
    if "whoami /groups" in lowered:
        return "T1033"
    if "wmic /node" in lowered:
        return "T1047"

    return merged_labels


def update_rare_labels(df, mapping=RARE_LABEL_MAPPING):
    """
    Hàm tự động cập nhật nhãn cho dataframe dựa trên từ điển ánh xạ (giảm nhãn từ notebook 01).
    """
    for old_label, new_label in mapping.items():
        df.loc[df["Labels"] == old_label, "Labels"] = new_label
    return df


def is_valid_label_string(label_str):
    """Kiểm tra chuỗi chỉ chứa parent technique IDs dạng Txxxx."""
    labels = [label.strip() for label in str(label_str).split(",")]
    return bool(labels) and all(re.fullmatch(r"T\d{4}", label) for label in labels)


def preprocess_dataframe(df_raw):
    """
    Chạy toàn bộ pipeline notebook 02 trên DataFrame raw.

    Kết quả đã được normalize text/labels, bỏ dòng rỗng và exact duplicates,
    merge những Cleaned_Text trùng nhau, resolve hard conflicts, sau đó tạo
    Tokenized_Text ở bước cuối cùng.
    """
    required_columns = {"Cleaned_Text", "Labels"}
    missing_columns = required_columns.difference(df_raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    processed = df_raw[["Cleaned_Text", "Labels"]].copy()
    processed["Cleaned_Text"] = processed["Cleaned_Text"].apply(normalize_cti_text)
    processed["Labels"] = processed["Labels"].apply(normalize_labels)
    processed = update_rare_labels(processed, RARE_LABEL_MAPPING)

    processed = processed[
        processed["Cleaned_Text"].str.strip().ne("")
        & processed["Labels"].str.strip().ne("")
    ].copy()
    processed = processed.drop_duplicates(
        subset=["Cleaned_Text", "Labels"]
    ).reset_index(drop=True)

    merged = (
        processed
        .groupby("Cleaned_Text", as_index=False)
        .agg({"Labels": merge_label_strings})
    )
    merged["Labels"] = merged.apply(
        lambda row: manual_resolve_hard_conflict(
            row["Cleaned_Text"], row["Labels"]
        ),
        axis=1,
    )
    merged = update_rare_labels(merged, RARE_LABEL_MAPPING)

    invalid_mask = ~merged["Labels"].apply(is_valid_label_string)
    if invalid_mask.any():
        examples = merged.loc[invalid_mask, "Labels"].head(10).tolist()
        raise ValueError(f"Invalid normalized labels detected: {examples}")

    mitre_id_pattern = r"\bT\d{4}(?:\.\d{3})?\b"
    leakage_mask = merged["Cleaned_Text"].str.contains(
        mitre_id_pattern, case=False, regex=True, na=False
    )
    if leakage_mask.any():
        raise ValueError(
            f"MITRE technique IDs remain in {int(leakage_mask.sum())} text rows"
        )

    # Tokenize cuối cùng, sau mọi bước normalize/merge, để toàn nhóm dùng chung
    # cùng một biểu diễn token trước khi tự chọn vectorizer và model.
    merged["Tokenized_Text"] = merged["Cleaned_Text"].apply(tokenize_cti_text)
    empty_token_mask = merged["Tokenized_Text"].str.strip().eq("")
    if empty_token_mask.any():
        raise ValueError(
            f"Tokenization produced {int(empty_token_mask.sum())} empty rows"
        )

    return merged[
        ["Cleaned_Text", "Tokenized_Text", "Labels"]
    ].reset_index(drop=True)


def get_label_frequency(df):
    """Trả về Series tần suất sample của từng label, giảm dần."""
    counter = Counter()
    for label_str in df["Labels"]:
        counter.update(label_set(label_str))
    return pd.Series(counter, name="Sample_Count", dtype="int64").sort_values(
        ascending=False
    )


def split_stage_datasets(df_processed, threshold=DEFAULT_STAGE1_THRESHOLD):
    """
    Chia full processed dataset thành Stage 1 frequent và Stage 2 rare-related.

    Stage 1 chỉ giữ labels có frequency >= threshold. Stage 2 giữ nguyên toàn bộ
    labels của những samples chứa ít nhất một rare label, giống notebook 02.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    label_frequency = get_label_frequency(df_processed)
    frequent_labels = set(label_frequency[label_frequency >= threshold].index)
    rare_labels = set(label_frequency[label_frequency < threshold].index)

    def keep_frequent(label_str):
        return ",".join(
            label
            for label in str(label_str).split(",")
            if label.strip() in frequent_labels
        )

    def contains_rare(label_str):
        return bool(label_set(label_str).intersection(rare_labels))

    stage1 = df_processed.copy()
    stage1["Labels"] = stage1["Labels"].apply(keep_frequent)
    stage1 = stage1[stage1["Labels"].str.len() > 0].reset_index(drop=True)

    stage2 = df_processed[
        df_processed["Labels"].apply(contains_rare)
    ].copy().reset_index(drop=True)

    return stage1, stage2, label_frequency


def run_preprocessing_pipeline(
    input_path,
    output_dir,
    threshold=DEFAULT_STAGE1_THRESHOLD,
):
    """Đọc raw CSV, chạy preprocessing, chia stages và ghi ba CSV đầu ra."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    processed = preprocess_dataframe(raw)
    stage1, stage2, label_frequency = split_stage_datasets(
        processed, threshold=threshold
    )

    processed_path = output_dir / "attack_dataset_processed.csv"
    stage1_path = output_dir / "attack_dataset_stage1_frequent.csv"
    stage2_path = output_dir / "attack_dataset_stage2_rare.csv"

    processed.to_csv(processed_path, index=False, encoding="utf-8")
    stage1.to_csv(stage1_path, index=False, encoding="utf-8")
    stage2.to_csv(stage2_path, index=False, encoding="utf-8")

    summary = {
        "input_rows": int(len(raw)),
        "processed_rows": int(len(processed)),
        "output_columns": processed.columns.tolist(),
        "total_labels": int(len(label_frequency)),
        "stage1_threshold": int(threshold),
        "stage1_rows": int(len(stage1)),
        "stage1_labels": int((label_frequency >= threshold).sum()),
        "stage2_rows": int(len(stage2)),
        "stage2_rare_labels": int((label_frequency < threshold).sum()),
        "processed_path": str(processed_path),
        "stage1_path": str(stage1_path),
        "stage2_path": str(stage2_path),
    }
    return summary


def build_argument_parser():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Preprocess CTI text and create full/Stage1/Stage2 datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "dataset" / "raw" / "New_Attack_Dataset.csv",
        help="Raw CSV containing Cleaned_Text and Labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset" / "processed",
        help="Directory for processed CSV files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_STAGE1_THRESHOLD,
        help="Minimum full-dataset label frequency for Stage 1 (default: 30).",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    summary = run_preprocessing_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
