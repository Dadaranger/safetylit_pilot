# SafetyLit PDF Scraper - Enhanced Version

A robust web scraping tool for extracting article metadata from SafetyLit weekly PDF bulletins with comprehensive quality control, incremental updates, and detailed reporting.

## Features

### Core Functionality
- Extracts article metadata (title, authors, journal, year) from SafetyLit weekly PDFs
- Supports scraping by month (January-August 2024)
- Exports data in both CSV and JSONL formats
- Comprehensive data validation and quality reporting

### Enhanced Features (New)

#### 1. Incremental Updates
- **State Tracking**: SQLite database tracks processed PDFs and entries
- **Smart Skipping**: Automatically skips already-processed content
- **Resume Capability**: Safely resume interrupted scraping sessions
- **Duplicate Prevention**: Entry-level duplicate detection prevents reprocessing

#### 2. Quality Control System
- **Pandas Integration**: Advanced duplicate detection and data standardization
- **Missing Value Analysis**: Comprehensive analysis of data completeness
- **Text Standardization**: Consistent formatting for titles, journals, and authors
- **Quality Scoring**: Automated quality assessment (0-100% scale)

#### 3. Comprehensive Logging & Reporting
- **Structured Logging**: File and console logging with detailed timestamps
- **Issue Tracking**: Database-backed issue logging with severity levels
- **Quality Reports**: Detailed JSON reports on data quality metrics
- **Comprehensive Reports**: Full documentation of challenges, solutions, and recommendations

#### 4. Error Handling & Reliability
- **Retry Mechanisms**: Exponential backoff for failed PDF downloads
- **SSL Handling**: Robust SSL certificate issue management
- **Graceful Degradation**: Continue processing even when individual PDFs fail
- **Performance Monitoring**: Track processing statistics and performance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Scrape January 2024 PDFs
python safetylit_pilot.py --month 1

# List available PDFs without scraping
python safetylit_pilot.py --month 1 --list
```

### Enhanced Usage with New Features

#### Enable Incremental Updates
```bash
# First run - processes all PDFs
python safetylit_pilot.py --month 1 --incremental

# Second run - skips already processed content
python safetylit_pilot.py --month 1 --incremental
```

#### Enable Quality Control
```bash
# Run with comprehensive quality control and standardization
python safetylit_pilot.py --month 1 --quality-control
```

#### Combined Enhanced Features
```bash
# Run with all enhancements
python safetylit_pilot.py --month 1 --incremental --quality-control --drop-noise
```

#### Check Processing Statistics
```bash
# View statistics from previous runs
python safetylit_pilot.py --stats --incremental
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--month 1-8` | Target month (January-August 2024) |
| `--list` | List available PDFs without scraping |
| `--incremental` | Enable incremental updates with state tracking |
| `--quality-control` | Enable comprehensive quality control |
| `--drop-noise` | Remove low-quality entries (headers, short abstracts) |
| `--keep-only` | Keep only entries with keep_flag=True |
| `--stats` | Show processing statistics (requires --incremental) |

## Output Files

### Standard Outputs
- `safetylit_pdf_2024_m01.csv` - Main dataset in CSV format
- `safetylit_pdf_2024_m01.jsonl` - Main dataset in JSONL format
- `validation_report.json` - Data validation summary

### Enhanced Outputs (New)
- `safetylit_pdf_2024_m01_quality_report.json` - Quality control metrics
- `safetylit_pdf_2024_m01_comprehensive_report.json` - Full analysis report
- `scraping_state.db` - SQLite database for incremental updates
- `safetylit_scraper.log` - Detailed execution logs

## Data Quality Features

### Duplicate Detection
- **Multi-level Detection**: Removes duplicates within PDFs and across months
- **Intelligent Matching**: Uses normalized titles, years, and journals
- **Pandas Integration**: Leverages pandas for sophisticated duplicate analysis

### Missing Value Analysis
- **Field Completeness**: Tracks completeness for critical fields (title, journal, year, authors)
- **Quality Scoring**: Automated quality assessment based on data completeness
- **Imputation Strategies**: Flags missing values for manual review or automated handling

### Data Standardization
- **Text Normalization**: Consistent capitalization and punctuation
- **Date Standardization**: Uniform year format extraction
- **Journal Name Standardization**: Title case formatting and punctuation cleanup
- **Quote Standardization**: Consistent quote character usage

## State Management

The enhanced scraper uses SQLite for state management:

### Database Tables
- `processed_pdfs`: Tracks processed PDF URLs with metadata
- `processed_entries`: Individual article entries with content hashes
- `scraping_issues`: Logs parsing errors and issues with severity levels

### Benefits
- **Resume Capability**: Restart interrupted sessions without losing progress
- **Performance**: Skip already-processed content for faster subsequent runs
- **Reliability**: Track and report processing issues for debugging
- **Analytics**: Comprehensive statistics on processing performance

## Quality Reports

### Quality Control Report
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "original_records": 1500,
  "cleaned_records": 1450,
  "records_removed": 50,
  "quality_score": 92.5,
  "missing_value_analysis": {
    "title": {"missing_count": 5, "missing_percentage": 0.3},
    "journal": {"missing_count": 120, "missing_percentage": 8.3}
  }
}
```

### Comprehensive Report
Includes:
- Processing statistics and performance metrics
- Documented challenges and implemented solutions
- Quality improvement recommendations
- Issue tracking and resolution status

## Error Handling

### Robust PDF Processing
- **Retry Logic**: Exponential backoff for network issues
- **SSL Workarounds**: Handles certificate validation problems
- **Graceful Failures**: Continue processing when individual PDFs fail
- **Issue Logging**: Comprehensive error tracking and categorization

### Common Issues and Solutions
1. **SSL Certificate Errors**: Automatically handled with verification bypass
2. **PDF Format Variations**: Robust parsing with multiple fallback strategies
3. **Network Timeouts**: Automatic retry with increasing delays
4. **Memory Management**: Efficient processing of large PDF collections

## Performance Optimizations

### Incremental Processing
- Skip already-processed PDFs based on content hash
- Entry-level duplicate detection prevents reprocessing
- Database indexing for fast lookup operations

### Memory Efficiency
- Stream processing for large datasets
- Pandas optimization for duplicate detection
- Efficient JSON serialization for reports

## Development Notes

### Architecture
- **Modular Design**: Separate classes for state management and quality control
- **Extensible**: Easy to add new quality checks and validations
- **Configurable**: Comprehensive command-line options for different use cases

### Testing Recommendations
1. Start with `--list` to verify PDF availability
2. Test with small month first (e.g., `--month 8` for August)
3. Enable `--incremental` for production runs
4. Use `--quality-control` for comprehensive data validation

### Future Enhancements
- Parallel PDF processing for improved performance
- Machine learning-based duplicate detection
- Real-time monitoring dashboard
- Schema validation for early error detection

## Troubleshooting

### Common Issues
1. **Import Error for pandas**: Install with `pip install pandas>=1.5.0`
2. **SSL Certificate Errors**: Already handled automatically
3. **Memory Issues**: Process smaller date ranges or enable incremental mode
4. **Database Locked**: Ensure no other instances are running

### Getting Help
- Check the log file: `safetylit_scraper.log`
- Review the comprehensive report for detailed analysis
- Use `--stats` to check processing statistics
- Enable `--quality-control` for detailed quality metrics

## License

This enhanced version builds upon the original SafetyLit scraper with significant improvements for production use in academic research environments.

## How to run this script for classmates

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Dadaranger/safetylit_pilot.git
cd safetylit_pilot

# Install dependencies
pip install -r requirements.txt

# Run basic scraping (e.g., January 2024)
python safetylit_pilot.py --month 1

# Run with all enhancements
python safetylit_pilot.py --month 1 --incremental --quality-control
```