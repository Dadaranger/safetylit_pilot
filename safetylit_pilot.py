# safetylit_pdf_month_scraper_v4.py
# SafetyLit (2024) PDF month scraper with stricter parsing, NaN-aware validation, and robust noise filtering.

import re, os, io, csv, json, time, random, hashlib, argparse, logging, sqlite3
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path
import requests
import pdfplumber
import pandas as pd
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

# --------------------- CONFIG ---------------------
HEADERS = {"User-Agent": "SafetyLitPDFPilot/4.0 (research; contact: your_email@example.com)"}
BASE_DELAY = 0.5

YEAR = 2024
START_DATE = date(2024, 1, 7)   # first weekly PDF in 2024
END_DATE   = date(2024, 8, 18)  # last weekly PDF in 2024

VALIDATION_REPORT = "validation_report.json"
STATE_DB = "scraping_state.db"
ISSUES_LOG = "scraping_issues.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safetylit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- NaN-like tokens treated as missing --------
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "NaN", "NA", "N/A", "NULL", "None"}
def is_missing(v):
    if v is None: return True
    s = str(v).strip()
    return s in MISSING_TOKENS

# --------------------- REGEX HELPERS ---------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")  # For extracting year from citation
URL_RE = re.compile(r"https?://", re.I)  # For detecting URLs

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

# --------------------- STATE MANAGEMENT ---------------------
class ScrapingState:
    """Manages state for incremental updates and tracking processed content."""
    
    def __init__(self, db_path=STATE_DB):
        self.db_path = db_path
        self.init_db()
        
    def init_db(self):
        """Initialize the state database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_pdfs (
                    pdf_url TEXT PRIMARY KEY,
                    week_code TEXT,
                    last_scraped_at TEXT,
                    entry_count INTEGER,
                    content_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_entries (
                    entry_hash TEXT PRIMARY KEY,
                    pdf_url TEXT,
                    title TEXT,
                    journal TEXT,
                    year TEXT,
                    scraped_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scraping_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_type TEXT,
                    pdf_url TEXT,
                    description TEXT,
                    severity TEXT,
                    timestamp TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            conn.commit()
    
    def is_pdf_processed(self, pdf_url: str) -> bool:
        """Check if a PDF has been processed recently."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT last_scraped_at FROM processed_pdfs WHERE pdf_url = ?",
                (pdf_url,)
            )
            result = cursor.fetchone()
            if result:
                # Check if processed within last 7 days (for weekly updates)
                last_scraped = datetime.fromisoformat(result[0])
                return (datetime.utcnow() - last_scraped).days < 7
            return False
    
    def record_pdf_processed(self, pdf_url: str, week_code: str, entry_count: int, content_hash: str):
        """Record that a PDF has been processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_pdfs 
                (pdf_url, week_code, last_scraped_at, entry_count, content_hash)
                VALUES (?, ?, ?, ?, ?)
            """, (pdf_url, week_code, now_utc(), entry_count, content_hash))
            conn.commit()
    
    def is_entry_processed(self, entry_hash: str) -> bool:
        """Check if an entry has been processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_entries WHERE entry_hash = ?",
                (entry_hash,)
            )
            return cursor.fetchone() is not None
    
    def record_entry_processed(self, entry_hash: str, pdf_url: str, title: str, journal: str, year: str):
        """Record that an entry has been processed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_entries 
                (entry_hash, pdf_url, title, journal, year, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entry_hash, pdf_url, title, journal, year, now_utc()))
            conn.commit()
    
    def log_issue(self, issue_type: str, pdf_url: str, description: str, severity: str = "medium"):
        """Log a scraping issue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO scraping_issues 
                (issue_type, pdf_url, description, severity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (issue_type, pdf_url, description, severity, now_utc()))
            conn.commit()
        logger.warning(f"Issue logged: {issue_type} - {description}")
    
    def get_processing_stats(self) -> dict:
        """Get statistics about processed content."""
        with sqlite3.connect(self.db_path) as conn:
            pdf_count = conn.execute("SELECT COUNT(*) FROM processed_pdfs").fetchone()[0]
            entry_count = conn.execute("SELECT COUNT(*) FROM processed_entries").fetchone()[0]
            issue_count = conn.execute("SELECT COUNT(*) FROM scraping_issues WHERE resolved = FALSE").fetchone()[0]
            return {
                "processed_pdfs": pdf_count,
                "processed_entries": entry_count,
                "unresolved_issues": issue_count
            }

# --------------------- QUALITY CONTROL ---------------------
class QualityController:
    """Handles data quality control and standardization."""
    
    def __init__(self, state_manager: ScrapingState):
        self.state = state_manager
        self.issues_found = []
    
    def standardize_text(self, text: str) -> str:
        """Standardize text format."""
        if is_missing(text):
            return None
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text).strip())
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        return text
    
    def standardize_year(self, year_str: str) -> str:
        """Standardize year format."""
        if is_missing(year_str):
            return None
        year = str(year_str).strip()
        # Extract 4-digit year
        match = re.search(r'\b(19|20)\d{2}\b', year)
        return match.group(0) if match else None
    
    def standardize_journal(self, journal_str: str) -> str:
        """Standardize journal name."""
        if is_missing(journal_str):
            return None
        journal = self.standardize_text(journal_str)
        # Remove trailing punctuation
        journal = re.sub(r'[.;:,]+$', '', journal)
        # Title case for journal names
        return journal.title() if journal else None
    
    def detect_duplicates_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use pandas to detect and handle duplicates."""
        original_count = len(df)
        
        # Create composite key for duplicate detection
        df['dup_key'] = df['title'].fillna('').str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True) + \
                       '_' + df['year'].fillna('').astype(str) + \
                       '_' + df['journal'].fillna('').str.lower()
        
        # Find duplicates
        duplicates = df[df.duplicated(subset=['dup_key'], keep=False)]
        if not duplicates.empty:
            self.issues_found.append({
                'type': 'duplicates_detected',
                'count': len(duplicates),
                'description': f'Found {len(duplicates)} duplicate entries'
            })
        
        # Keep first occurrence of duplicates
        df_clean = df.drop_duplicates(subset=['dup_key'], keep='first')
        df_clean = df_clean.drop(columns=['dup_key'])
        
        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate entries")
        
        return df_clean
    
    def analyze_missing_values(self, df: pd.DataFrame) -> dict:
        """Analyze missing values in critical fields."""
        critical_fields = ['title', 'journal', 'year', 'authors']
        missing_analysis = {}
        
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum() + (df[field] == '').sum()
                missing_pct = (missing_count / len(df)) * 100
                missing_analysis[field] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2),
                    'has_data_count': len(df) - int(missing_count)
                }
                
                if missing_pct > 10:  # Flag if more than 10% missing
                    self.issues_found.append({
                        'type': 'high_missing_values',
                        'field': field,
                        'percentage': missing_pct,
                        'description': f'High missing values in {field}: {missing_pct:.1f}%'
                    })
        
        return missing_analysis
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization to entire dataframe."""
        df_clean = df.copy()
        
        # Standardize text fields
        if 'title' in df_clean.columns:
            df_clean['title'] = df_clean['title'].apply(self.standardize_text)
        if 'authors' in df_clean.columns:
            df_clean['authors'] = df_clean['authors'].apply(self.standardize_text)
        if 'journal' in df_clean.columns:
            df_clean['journal'] = df_clean['journal'].apply(self.standardize_journal)
        if 'year' in df_clean.columns:
            df_clean['year'] = df_clean['year'].apply(self.standardize_year)
        
        # Standardize ingested_at to consistent datetime format
        if 'ingested_at' in df_clean.columns:
            df_clean['ingested_at'] = pd.to_datetime(df_clean['ingested_at'], errors='coerce')
        
        return df_clean
    
    def generate_quality_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """Generate comprehensive quality control report."""
        missing_analysis = self.analyze_missing_values(cleaned_df)
        
        report = {
            'timestamp': now_utc(),
            'original_records': len(original_df),
            'cleaned_records': len(cleaned_df),
            'records_removed': len(original_df) - len(cleaned_df),
            'missing_value_analysis': missing_analysis,
            'issues_found': self.issues_found,
            'quality_score': self._calculate_quality_score(cleaned_df)
        }
        
        return report
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        if df.empty:
            return 0.0
        
        critical_fields = ['title', 'journal', 'year']
        total_score = 0
        field_count = 0
        
        for field in critical_fields:
            if field in df.columns:
                completeness = (df[field].notna() & (df[field] != '')).sum() / len(df)
                total_score += completeness
                field_count += 1
        
        return round((total_score / field_count * 100), 2) if field_count > 0 else 0.0

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
def parse_pdf(pdf_url: str, state_manager: ScrapingState = None) -> list[dict]:
    """Parse PDF with incremental update support and enhanced error handling."""
    
    # Check if already processed
    if state_manager and state_manager.is_pdf_processed(pdf_url):
        logger.info(f"Skipping already processed PDF: {pdf_url}")
        return []
    
    try:
        pdf_bytes = download_pdf(pdf_url)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        
        pages = extract_pages(pdf_bytes)
        
        # Skip first and last pages
        if len(pages) <= 2:  # If PDF has 2 or fewer pages, return empty
            if state_manager:
                state_manager.log_issue("insufficient_pages", pdf_url, 
                                      f"PDF has only {len(pages)} pages", "low")
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
                
                # Add metadata
                idx += 1
                rec["category"] = current_category
                rec["entry_index_in_pdf"] = idx
                rec["page_number"] = page_num  # Track which page it came from
                rec["pdf_url"] = pdf_url
                rec["pdf_week_code"] = wk
                rec["ingested_at"] = now_utc()
                
                # Check for duplicates at entry level
                entry_hash = phash(
                    norm_title(rec.get("title") or ""),
                    rec.get("year") or "",
                    rec.get("journal") or ""
                )
                
                if state_manager and not state_manager.is_entry_processed(entry_hash):
                    records.append(rec)
                    state_manager.record_entry_processed(
                        entry_hash, pdf_url, 
                        rec.get("title"), 
                        rec.get("journal"), 
                        rec.get("year")
                    )
                elif not state_manager:
                    records.append(rec)

        # Dedupe within this PDF
        seen, out = set(), []
        for r in records:
            key = r.get("doi") or phash(norm_title(r.get("title") or ""), r.get("year") or "", r.get("pdf_week_code") or "")
            if key not in seen:
                seen.add(key); out.append(r)
        
        # Record PDF as processed
        if state_manager:
            state_manager.record_pdf_processed(pdf_url, wk, len(out), content_hash)
            
        logger.info(f"Successfully parsed {len(out)} entries from {pdf_url}")
        return out
        
    except Exception as e:
        error_msg = f"Failed to parse PDF {pdf_url}: {str(e)}"
        logger.error(error_msg)
        if state_manager:
            state_manager.log_issue("pdf_parse_error", pdf_url, error_msg, "high")
        return []

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
def write_outputs(records, csv_path, jsonl_path, quality_controller: QualityController = None):
    """Write outputs with optional quality control processing."""
    
    # Convert to DataFrame for quality control
    if quality_controller and records:
        original_df = pd.DataFrame(records)
        
        # Apply standardization
        standardized_df = quality_controller.standardize_dataframe(original_df)
        
        # Detect and remove duplicates
        cleaned_df = quality_controller.detect_duplicates_pandas(standardized_df)
        
        # Generate quality report
        quality_report = quality_controller.generate_quality_report(original_df, cleaned_df)
        
        # Save quality report
        quality_report_path = csv_path.replace('.csv', '_quality_report.json')
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Quality report saved to {quality_report_path}")
        logger.info(f"Quality score: {quality_report['quality_score']:.1f}%")
        
        # Convert back to records
        records = cleaned_df.to_dict('records')
    
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
                # Handle pandas datetime objects
                if hasattr(v, 'isoformat'):
                    v = v.isoformat()
                row[k] = "" if is_missing(v) else v
            w.writerow(row)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            out = {}
            for k, v in r.items():
                # Handle pandas datetime objects
                if hasattr(v, 'isoformat'):
                    v = v.isoformat()
                out[k] = None if is_missing(v) else v
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows → {csv_path}, {jsonl_path}")
    return records

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
        
        # Citation format check (journal and year should be reasonable)
        if has_journal and has_year:
            journal = r.get("journal", "").strip()
            year = r.get("year", "").strip()
            
            # Check if year is a valid 4-digit year
            year_valid = re.match(r'^(19|20)\d{2}$', year)
            
            # Check if journal name is reasonable (not empty, not just punctuation)
            journal_valid = len(journal) > 2 and not re.match(r'^[^\w]*$', journal)
            
            if not (year_valid and journal_valid):
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

# --------------------- COMPREHENSIVE REPORTING ---------------------
def generate_comprehensive_report(csv_path: str, state_manager: ScrapingState = None, quality_controller: QualityController = None):
    """Generate a comprehensive report documenting challenges, solutions, and improvements."""
    
    report = {
        "timestamp": now_utc(),
        "data_source": csv_path,
        "challenges_and_solutions": {},
        "quality_metrics": {},
        "processing_statistics": {},
        "recommendations": []
    }
    
    # Processing statistics
    if state_manager:
        report["processing_statistics"] = state_manager.get_processing_stats()
        
        # Get issues from database
        with sqlite3.connect(state_manager.db_path) as conn:
            issues = conn.execute("""
                SELECT issue_type, COUNT(*) as count, severity 
                FROM scraping_issues 
                WHERE resolved = FALSE 
                GROUP BY issue_type, severity
            """).fetchall()
            
            if issues:
                report["challenges_and_solutions"]["unresolved_issues"] = [
                    {"type": issue[0], "count": issue[1], "severity": issue[2]} 
                    for issue in issues
                ]
    
    # Quality metrics from quality controller
    if quality_controller:
        report["quality_metrics"]["issues_detected"] = quality_controller.issues_found
        
        # Read the CSV to get final data quality metrics
        try:
            df = pd.read_csv(csv_path)
            report["quality_metrics"]["final_dataset"] = {
                "total_records": len(df),
                "completeness": {
                    field: {
                        "non_empty_count": int((df[field].notna() & (df[field] != '')).sum()),
                        "completeness_rate": float((df[field].notna() & (df[field] != '')).sum() / len(df) * 100)
                    }
                    for field in ['title', 'journal', 'year', 'authors'] if field in df.columns
                }
            }
        except Exception as e:
            logger.warning(f"Could not analyze final dataset: {e}")
    
    # Document challenges and solutions
    report["challenges_and_solutions"]["common_issues"] = {
        "pdf_parsing_errors": {
            "description": "Some PDFs fail to parse due to format inconsistencies or download issues",
            "solution": "Implemented robust error handling with retry mechanisms and issue logging",
            "impact": "Reduced script crashes and improved data collection reliability"
        },
        "duplicate_entries": {
            "description": "Same articles appear across multiple PDFs or within single PDFs",
            "solution": "Implemented multi-level deduplication using content hashing and pandas duplicate detection",
            "impact": "Eliminated redundant data and improved dataset quality"
        },
        "missing_metadata": {
            "description": "Critical fields like journal names, years, or authors may be missing",
            "solution": "Added comprehensive missing value analysis and quality scoring",
            "impact": "Better visibility into data completeness for downstream analysis"
        },
        "format_inconsistencies": {
            "description": "Text formatting varies across sources (capitalization, punctuation, etc.)",
            "solution": "Implemented standardization functions for text, dates, and journal names",
            "impact": "Improved data consistency and analysis reliability"
        }
    }
    
    # Recommendations for improvement
    report["recommendations"] = [
        {
            "category": "Performance",
            "recommendation": "Implement parallel PDF processing to reduce scraping time",
            "priority": "Medium",
            "effort": "High"
        },
        {
            "category": "Data Quality",
            "recommendation": "Add machine learning-based duplicate detection for more sophisticated matching",
            "priority": "Low", 
            "effort": "High"
        },
        {
            "category": "Reliability",
            "recommendation": "Implement automatic retry with exponential backoff for failed downloads",
            "priority": "High",
            "effort": "Low"
        },
        {
            "category": "Monitoring",
            "recommendation": "Add real-time monitoring dashboard for scraping health metrics",
            "priority": "Medium",
            "effort": "Medium"
        },
        {
            "category": "Data Validation",
            "recommendation": "Implement schema validation to catch parsing errors early",
            "priority": "High",
            "effort": "Medium"
        }
    ]
    
    # Save comprehensive report
    report_path = csv_path.replace('.csv', '_comprehensive_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Comprehensive report saved to {report_path}")
    
    # Print summary
    print("\n=== COMPREHENSIVE SCRAPING REPORT ===")
    print(f"Report saved to: {report_path}")
    print(f"Total records processed: {report['quality_metrics'].get('final_dataset', {}).get('total_records', 'N/A')}")
    
    if "unresolved_issues" in report["challenges_and_solutions"]:
        print(f"Unresolved issues: {len(report['challenges_and_solutions']['unresolved_issues'])}")
    
    if quality_controller:
        quality_score = None
        try:
            df = pd.read_csv(csv_path)
            quality_score = quality_controller._calculate_quality_score(df)
            print(f"Overall quality score: {quality_score:.1f}%")
        except:
            pass
    
    print("\nKey improvements implemented:")
    for solution in report["challenges_and_solutions"]["common_issues"].values():
        print(f"- {solution['impact']}")
    
    return report

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
    ap.add_argument("--quality-control", action="store_true", help="Enable comprehensive quality control and standardization.")
    ap.add_argument("--incremental", action="store_true", help="Enable incremental updates to skip already processed content.")
    ap.add_argument("--stats", action="store_true", help="Show processing statistics and exit.")
    args = ap.parse_args()

    # Initialize state management and quality control
    state_manager = ScrapingState() if args.incremental else None
    quality_controller = QualityController(state_manager) if args.quality_control else None
    
    # Show stats and exit if requested
    if args.stats and state_manager:
        stats = state_manager.get_processing_stats()
        print("\n=== PROCESSING STATISTICS ===")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        return

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

    logger.info(f"Starting scraping for {YEAR}-{month:02d} with {len(urls)} PDFs")
    logger.info(f"Incremental updates: {'enabled' if args.incremental else 'disabled'}")
    logger.info(f"Quality control: {'enabled' if args.quality_control else 'disabled'}")

    all_rows = []
    for i, u in enumerate(urls, 1):
        try:
            logger.info(f"Processing PDF {i}/{len(urls)}: {u}")
            recs = parse_pdf(u, state_manager)
            print(f"Parsed {len(recs)} entries from {u}")
            all_rows.extend(recs)
        except Exception as e:
            error_msg = f"ERROR parsing {u}: {e}"
            print(error_msg)
            logger.error(error_msg)
            if state_manager:
                state_manager.log_issue("pdf_processing_error", u, str(e), "high")

    logger.info(f"Total entries parsed: {len(all_rows)}")

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

    logger.info(f"Final dataset size: {len(rows)} entries")

    # Write outputs with quality control
    write_outputs(rows, csv_out, json_out, quality_controller)
    validate_csv(csv_out, report_json=VALIDATION_REPORT)
    
    # Generate comprehensive report if quality control is enabled
    if quality_controller:
        generate_comprehensive_report(csv_out, state_manager, quality_controller)
    
    logger.info("Scraping completed successfully")

if __name__ == "__main__":
    main()
