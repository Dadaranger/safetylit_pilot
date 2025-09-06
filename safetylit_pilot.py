# safetylit_pdf_month_scraper_v4.py
# SafetyLit (2024) PDF month scraper with stricter parsing, NaN-aware validation, and robust noise filtering.

import re, os, io, csv, json, time, random, hashlib, argparse
from datetime import datetime, date, timedelta
from collections import defaultdict
import requests
import pdfplumber
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

# --------------------- CONFIG ---------------------
HEADERS = {"User-Agent": "SafetyLitPDFPilot/4.0 (research; contact: your_email@example.com)"}
BASE_DELAY = 0.5

YEAR = 2024
START_DATE = date(2024, 1, 7)   # first weekly PDF in 2024
END_DATE   = date(2024, 8, 18)  # last weekly PDF in 2024

VALIDATION_REPORT = "validation_report.json"

# -------- NaN-like tokens treated as missing --------
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "NaN", "NA", "N/A", "NULL", "None"}
def is_missing(v):
    if v is None: return True
    s = str(v).strip()
    return s in MISSING_TOKENS

# --------------------- REGEX HELPERS ---------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")  # For extracting year from citation

# ---- Noise patterns (headers, orgs, chrome) ----
BULLETIN_NOISE_PATTERNS = [
    re.compile(r"^\s*SafetyLit\b", re.I),
    re.compile(r"Weekly Literature Update Bulletin", re.I),
    re.compile(r"in collaboration with", re.I),
    re.compile(r"San Diego State University", re.I),
    re.compile(r"copyright", re.I),
    re.compile(r"All rights reserved", re.I),
    re.compile(r"ISSN", re.I),
    re.compile(r"World Health Organization\.?", re.I),
    re.compile(r"United Nations(?:\.|$)", re.I),
]
TITLE_BLACKLIST = [
    re.compile(r"^\s*World Health Organization\.?\s*$", re.I),
    re.compile(r"^\s*United Nations\.?\s*$", re.I),
    re.compile(r"^\s*Weekly Literature Update Bulletin", re.I),
    re.compile(r"^\s*SafetyLit\b", re.I)
]
SECTION_LINE_HINT = re.compile(r"^[A-Z0-9][A-Z0-9 \-\(\)&/]{5,}$")  # ALL-CAPS-ish headings

# --------------------- NET ---------------------
# Configure requests to handle SSL issues
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.headers.update(HEADERS)
session.verify = False  # Disable SSL verification due to certificate issues

@retry(wait=wait_exponential_jitter(initial=1, max=15), stop=stop_after_attempt(4))
def download_pdf(url: str) -> bytes:
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        time.sleep(BASE_DELAY + random.random()*0.4)
        return r.content
    except requests.exceptions.SSLError as e:
        print(f"SSL Error downloading {url}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise

# --------------------- UTIL ---------------------
def now_utc() -> str:
    return datetime.utcnow().isoformat()

def week_code_from_date(d: date) -> str:
    return d.strftime("%y%m%d")

def week_url_from_date(d: date) -> str:
    return f"https://www.safetylit.org/week/{d.year}/{week_code_from_date(d)}.pdf"

def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def phash(*parts) -> str:
    return hashlib.sha1("||".join([p or "" for p in parts]).encode("utf-8")).hexdigest()

# --------------------- DATE RANGE ---------------------
def weekly_sundays_for_month(year: int, month: int) -> list[date]:
    first = date(year, month, 1)
    last  = (date(year, month+1, 1) - timedelta(days=1)) if month < 12 else date(year, 12, 31)
    d = first
    while d.weekday() != 6:  # Sunday
        d += timedelta(days=1)
    out = []
    while d <= last:
        if START_DATE <= d <= END_DATE:
            out.append(d)
        d += timedelta(days=7)
    return out

# --------------------- TEXT EXTRACTION ---------------------
def extract_pages(pdf_bytes: bytes) -> list[list[str]]:
    """Return pages as list of cleaned lines (header/footer & noise removed)."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=1.5, y_tolerance=2.0) or ""
            lines = [re.sub(r"[ \t]+$", "", ln) for ln in txt.replace("\r", "\n").split("\n")]
            lines = [ln for ln in lines if ln.strip()]
            # drop footers like "Page 3 of 22"
            lines = [ln for ln in lines if not re.search(r"Page\s+\d+\s+of\s+\d+", ln, re.I)]
            # drop known chrome
            cleaned = []
            for ln in lines:
                if any(p.search(ln) for p in BULLETIN_NOISE_PATTERNS):
                    continue
                cleaned.append(ln)
            pages.append(cleaned)
    return pages

def split_blocks(lines: list[str]) -> list[list[str]]:
    """Split into blocks by blank-ish gaps."""
    blocks, buf = [], []
    for ln in lines:
        if not ln.strip():
            if buf: blocks.append(buf); buf = []
        else:
            buf.append(ln)
    if buf: blocks.append(buf)
    return blocks

# --------------------- PARSING ---------------------
def looks_like_title(s: str) -> bool:
    if len(s) < 8: return False
    if any(p.search(s) for p in TITLE_BLACKLIST): return False
    if any(p.search(s) for p in BULLETIN_NOISE_PATTERNS): return False
    if s.endswith(":"): return False
    return True

def looks_like_authors(s: str) -> bool:
    if sum(1 for c in s if c in {",", ";"}) >= 1 and len(s) < 300:
        return True
    if len(s.split()) >= 4 and any("." in tok for tok in s.split()):
        return True
    return False

def parse_source_line(blob: str) -> dict:
    out = {"source_line": blob or None}
    if is_missing(blob): return out

    # year
    year = None
    my = YEAR_RE.search(blob)
    if my: year = my.group(0)

    # journal
    journal = None
    if year:
        parts = blob.split(year, 1)
        if parts:
            jpart = parts[0].strip(" .;:-")
            # Check length is reasonable for journal name
            if 3 <= len(jpart) <= 180:
                journal = jpart
    else:
        cand = blob.split(";")[0].strip()
        if 3 <= len(cand) <= 180:
            journal = cand

    out.update({
        "journal": journal or None,
        "year": year or None
    })
    return out

def parse_citation_line(line: str) -> dict:
    """Parse the main citation line that follows pattern: 'Authors. Journal Year'"""
    result = {}
    
    # Split on the first period after authors
    parts = line.split(".", 1)
    if len(parts) != 2:
        return result
    
    result["authors"] = parts[0].strip()
    meta = parts[1].strip()
    
    # Extract journal name (everything before the year)
    m_year = YEAR_RE.search(meta)
    if m_year:
        journal = meta[:m_year.start()].strip(" ,;")
        if journal:
            result["journal"] = journal
        result["year"] = m_year.group(0)
        # No need to parse further details
    
    return result

def parse_block(block_lines: list[str]) -> dict:
    """Parse a block of text into an article entry."""
    if len(block_lines) < 2:  # Need at least title and citation
        return {}
        
    # Check if this is a category marker
    if len(block_lines) == 1 and SECTION_LINE_HINT.match(block_lines[0]):
        return {"_category_marker": block_lines[0].strip()}
    
    # Find the main citation line that starts with a dash and contains year
    citation_idx = None
    citation_parts = None
    
    for i, line in enumerate(block_lines):
        if line.strip().startswith("-") and YEAR_RE.search(line):
            clean_line = line.strip().lstrip("- ")
            citation_parts = parse_citation_line(clean_line)
            if citation_parts.get("journal") and citation_parts.get("year"):
                citation_idx = i
                break
    
    if citation_idx is None:
        return {}  # No valid citation line found
        
    # Title is everything before the citation line, joined
    title_lines = []
    for line in block_lines[:citation_idx]:
        line = line.strip()
        if not line or line.startswith("(Copyright"):
            continue
        title_lines.append(line)
    
    title = " ".join(title_lines).strip()
    if not title or len(title) < 8:
        return {}

    # Process remaining content (if needed in the future)
    pass

    # Validate required fields
    title_not_noise = not any(p.search(title) for p in TITLE_BLACKLIST)
    has_citation = bool(citation_parts.get("journal") and citation_parts.get("year"))
    keep = title_not_noise and has_citation

    return {
        "title": title,
        "authors": citation_parts.get("authors"),
        "journal": citation_parts.get("journal"),
        "year": citation_parts.get("year")
    }

# --------------------- PDF PARSE ---------------------
def parse_pdf(pdf_url: str) -> list[dict]:
    pdf_bytes = download_pdf(pdf_url)
    pages = extract_pages(pdf_bytes)
    
    # Skip first and last pages
    if len(pages) <= 2:  # If PDF has 2 or fewer pages, return empty
        return []
    pages = pages[1:-1]  # Use only middle pages

    records, current_category = [], None
    idx = 0
    wk = pdf_url.split("/")[-1].replace(".pdf", "")
    
    for page_num, lines in enumerate(pages, 1):  # 1-based page numbering
        blocks = split_blocks(lines)
        for blk in blocks:
            rec = parse_block(blk)
            if not rec:
                continue
            if "_category_marker" in rec:
                current_category = rec["_category_marker"]
                continue
            idx += 1
            rec["category"] = current_category
            rec["entry_index_in_pdf"] = idx
            rec["page_number"] = page_num  # Track which page it came from
            rec["pdf_url"] = pdf_url
            rec["pdf_week_code"] = wk
            rec["ingested_at"] = now_utc()
            records.append(rec)

    # Dedupe
    seen, out = set(), []
    for r in records:
        key = r.get("doi") or phash(norm_title(r.get("title") or ""), r.get("year") or "", r.get("pdf_week_code") or "")
        if key not in seen:
            seen.add(key); out.append(r)
    return out

# --------------------- NOISE FILTER (post-parse) ---------------------
def drop_noise_rows(rows):
    """Drop rows that still look like non-articles despite parsing."""
    cleaned = []
    for r in rows:
        t = (r.get("title") or "").strip()
        if any(p.search(t) for p in TITLE_BLACKLIST):
            continue
        # If abstract is very short and no meta at all, drop as noise.
        abstract = r.get("abstract") or ""
        meta_ok = any(not is_missing(r.get(k)) for k in ("journal","year","doi"))
        if len(abstract) < 30 and not meta_ok:
            continue
        # If journal looks like a URL, blank it out (don’t use as a keep signal)
        j = r.get("journal")
        if isinstance(j, str) and URL_RE.search(j):
            r["journal"] = None
        cleaned.append(r)
    return cleaned

# --------------------- OUTPUTS ---------------------
def write_outputs(records, csv_path, jsonl_path):
    fields = [
        "title","authors","journal","year",
        "pdf_url","pdf_week_code","page_number","entry_index_in_pdf","ingested_at"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            # normalize NaN-like tokens to empty strings in CSV
            row = {}
            for k in fields:
                v = r.get(k)
                row[k] = "" if is_missing(v) else v
            w.writerow(row)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            out = {k: (None if is_missing(v) else v) for k, v in r.items()}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows → {csv_path}, {jsonl_path}")

# --------------------- VALIDATION ---------------------
def validate_csv(csv_path, report_json=VALIDATION_REPORT):
    """Validate parsed SafetyLit entries according to expected format:
    Title - Authors. Journal Year; Volume(Issue): Pages.
    (Copyright Info)
    """
    with open(csv_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    n = len(rows)
    
    # Validation buckets
    incomplete_citation = []  # Missing essential citation parts
    malformed_citation = []  # Citation doesn't match expected format
    missing_copyright = []   # No copyright info
    dups = []               # Duplicate entries
    low_confidence = []     # Low parsing confidence scores
    seen = set()
    
    citation_stats = defaultdict(int)  # Track presence of citation components
    
    for r in rows:
        # Check citation completeness
        has_authors = not is_missing(r.get("authors"))
        has_journal = not is_missing(r.get("journal"))
        has_year = not is_missing(r.get("year"))
        
        citation_stats["has_authors"] += int(has_authors)
        citation_stats["has_journal"] += int(has_journal)
        citation_stats["has_year"] += int(has_year)
        # Essential citation parts check
        if not (has_authors and has_journal and has_year):
            incomplete_citation.append(r)
        
        # Citation format check (should be journal name followed by year)
        if has_journal and has_year:
            journal = r.get("journal", "").strip()
            year = r.get("year", "").strip()
            if not re.search(rf"{re.escape(journal)}.*{year}", r.get("source_line", "")):
                malformed_citation.append(r)
        
        # Duplicates (using title+year as key)
        key = f"{norm_title(r.get('title'))}__{r.get('year')}"
        if key in seen:
            dups.append(r)
        else:
            seen.add(key)
    
    def pct(x): return round(100.0 * x / n, 2) if n else 0.0
    
    rules = {
        "citation_complete": {
            "pass": n - len(incomplete_citation),
            "pass_rate_pct": pct(n - len(incomplete_citation)),
            "fail": len(incomplete_citation)
        },
        "citation_format": {
            "pass": n - len(malformed_citation),
            "pass_rate_pct": pct(n - len(malformed_citation)),
            "fail": len(malformed_citation)
        },
        "duplicates": {
            "duplicate_count": len(dups)
        }
    }
    
    # Add citation component stats
    stats = {
        "total_entries": n,
        "citation_components": {
            k: {"count": v, "percentage": pct(v)}
            for k, v in citation_stats.items()
        }
    }
    
    summary = {
        "rows": n,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "rules": rules,
        "stats": stats
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DATA VALIDATION SUMMARY ===")
    print(f"Rows: {n}")
    for k, v in rules.items():
        if "pass_rate_pct" in v:
            print(f"- {k}: {v['pass_rate_pct']}% pass (fails: {v['fail']})")
        else:
            print(f"- {k}: {v}")
    print(f"Report: {report_json}\n")
    return summary

# --------------------- CLI + MAIN ---------------------
def prompt_month() -> int:
    while True:
        s = input("Enter month number for 2024 (1–8): ").strip()
        if s.isdigit() and 1 <= int(s) <= 8:
            return int(s)
        print("Month must be between 1 and 8 (Jan..Aug for 2024).")

def main():
    ap = argparse.ArgumentParser(description="Scrape SafetyLit weekly PDFs (2024) for a selected month.")
    ap.add_argument("--month", type=int, help="Target month (1–8) within 2024 range")
    ap.add_argument("--list", action="store_true", help="Only list weekly URLs and exit.")
    ap.add_argument("--keep-only", action="store_true", help="Write only rows with keep_flag=True.")
    ap.add_argument("--drop-noise", action="store_true", help="Drop noise rows (date/volume/org headers; short abstract + no meta).")
    args = ap.parse_args()

    month = args.month if args.month is not None else prompt_month()
    if not (1 <= month <= 8): raise SystemExit("Month must be 1..8 for 2024 range.")

    sundays = weekly_sundays_for_month(YEAR, month)
    if not sundays:
        print(f"No weekly bulletins in {YEAR}-{month:02d} within allowed range."); return
    urls = [week_url_from_date(d) for d in sundays]
    print(f"\nTargeting {len(urls)} weekly PDFs for {YEAR}-{month:02d}:")
    for u in urls: print("  -", u)
    if args.list: return

    tag = f"{YEAR}_m{month:02d}"
    csv_out  = f"safetylit_pdf_{tag}.csv"
    json_out = f"safetylit_pdf_{tag}.jsonl"

    all_rows = []
    for u in urls:
        try:
            recs = parse_pdf(u)
            print(f"Parsed {len(recs)} entries from {u}")
            all_rows.extend(recs)
        except Exception as e:
            print(f"ERROR parsing {u}: {e}")

    # Dedupe across month
    seen, rows = set(), []
    for r in all_rows:
        key = r.get("doi") or phash(norm_title(r.get("title") or ""), r.get("year") or "", r.get("pdf_week_code") or "")
        if key not in seen:
            seen.add(key); rows.append(r)

    # Post-parse journal URL cleanup + noise drop option
    rows = drop_noise_rows(rows) if args.drop_noise else rows

    if args.keep_only:
        rows = [r for r in rows if r.get("keep_flag")]

    write_outputs(rows, csv_out, json_out)
    validate_csv(csv_out, report_json=VALIDATION_REPORT)

if __name__ == "__main__":
    main()
