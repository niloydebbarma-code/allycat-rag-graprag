import os, sys
import shutil
from pathlib import Path
from docling.document_converter import DocumentConverter
import logging
from my_config import MY_CONFIG

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

shutil.rmtree(MY_CONFIG.PROCESSED_DATA_DIR, ignore_errors=True)
shutil.os.makedirs(MY_CONFIG.PROCESSED_DATA_DIR, exist_ok=True)
logger.info (f"✅ Cleared  processed data directory :  {MY_CONFIG.PROCESSED_DATA_DIR}")

converter = DocumentConverter(format_options={"preserve_links": True})

input_path = Path(MY_CONFIG.CRAWL_DIR)
# input_files = list(input_path.glob('*.html')) + list(input_path.glob('*.htm')) + list(input_path.glob('*.pdf'))
input_files = list(input_path.glob('*.*')) 
logger.info (f"Found {len(input_files)} files to process in {input_path}")

files_processed = 0
errors = 0
for input_file in input_files:
    try:
        result = converter.convert(input_file)
        markdown_content = result.document.export_to_markdown()
        
        md_file_name = os.path.join(MY_CONFIG.PROCESSED_DATA_DIR, f"{input_file.stem}.md")
        with open(md_file_name, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
            
        logger.info(f"Converted '{input_file}' --> '{md_file_name}'")
        files_processed += 1
    except Exception as e:
        errors += 1
        logger.warning(f"Error processing {input_file}: {e}")

logger.info (f"✅ Processed {files_processed} files.  Errors: {errors}")