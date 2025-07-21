# Narrahunt Phase 2

## Directory Structure

This project follows a standard Python structure with the following directories:

- **core/**: Main system files and controllers
  - detective_agent.py: Central reasoning engine
  - artifact_extractor.py: Extracts Ethereum artifacts from content
  - crawler.py, fetch.py, url_queue.py: Web crawling components
  - llm_integration.py: LLM API integration
  - narrative_matrix.py, objectives_manager.py: Narrative discovery components
  - main.py, integrated_main.py: Main entry points

- **modules/**: Utility modules
  - run_validation.py: Testing and validation scripts
  - verify_api_keys.py: API key verification

- **config/**: Configuration files
  - JSON configuration files
  - Wordlists and other static data

- **docs/**: Documentation
  - README and other Markdown files
  - Implementation summaries
  - Bug fix documentation

- **tests/**: Test files
  - Unit tests
  - Integration tests
  - Validation scripts

- **backup/**: Backup files
  - Previous versions of critical files

- **temp/**: Temporary and generated files
  - Log files
  - Extracted data

## Additional Directories

- **logs/**: Log files from various components
- **results/**: Crawl and analysis results
- **enhancements/**: Enhanced functionality modules
- **cache/**: Cached data for performance

## Usage

To run the main crawler:

```bash
python core/main.py
```

To run the integrated controller:

```bash
python core/integrated_main.py
```

## Documentation

For more detailed information, see the files in the `docs/` directory.