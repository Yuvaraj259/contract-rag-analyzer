import os
import fitz  # PyMuPDF
import docx
import boto3
import tempfile
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io

# Load environment variables (from .env)
load_dotenv()

# Initialize the S3 client using credentials from the environment
s3_client = boto3.client('s3')

def load_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text += f"\n--- PAGE {page_num + 1} ---\n"
            page_text = page.get_text("text").strip()
            
            # If page text is very short, it might be a scanned image. Use OCR.
            if len(page_text) < 50:
                try:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        page_text = ocr_text.strip()
                except Exception as ocr_err:
                    print(f"OCR failed for page {page_num + 1}: {ocr_err}")
                    
            text += page_text + "\n"
            
            # Extract embedded hyperlinks
            links = page.get_links()
            if links:
                text += "\n--- Embedded Links ---\n"
                for link in links:
                    if 'uri' in link:
                        rect = link['from']
                        link_text = page.get_textbox(rect).strip().replace('\n', ' ')
                        if link_text:
                            text += f"- [{link_text}]({link['uri']})\n"
                        else:
                            text += f"- {link['uri']}\n"
                text += "\n"
                
        doc.close()
    except Exception as e:
        raise Exception(f"Failed to read PDF {file_path}: {e}")
    return text

def load_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        raise Exception(f"Failed to read DOCX {file_path}: {e}")
    return text

def download_from_s3(s3_path):
    """Downloads a file from S3 to a temporary local file and returns the local path."""
    # S3 path format expected: s3://bucket-name/file_key.pdf
    # or just use the bucket from .env if only key is provided
    bucket_name = os.getenv('S3_BUCKET_NAME')
    
    if s3_path.startswith("s3://"):
        # Extract bucket and key from s3://bucket/key
        parts = s3_path.replace("s3://", "").split("/", 1)
        if len(parts) == 2:
            bucket_name, s3_key = parts
        else:
            s3_key = parts[0]
    else:
        # Assume it's just the object key and use the default bucket
        s3_key = s3_path

    # Create a temporary file to hold the downloaded document
    ext = os.path.splitext(s3_key)[1]
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd) # Close file descriptor so boto3 can write to it
    
    try:
        print(f"Downloading {s3_key} from bucket {bucket_name}...")
        s3_client.download_file(bucket_name, s3_key, temp_path)
        return temp_path
    except Exception as e:
        os.remove(temp_path)
        raise Exception(f"Failed to download from S3: {e}")

def load_document(file_path):
    """Loads a document and returns its text content based on the file extension.
       Supports local files and AWS S3 files (e.g., s3://bucket/key.pdf).
    """
    is_s3 = file_path.startswith("s3://") or not os.path.exists(file_path)
    
    local_path = file_path
    if is_s3:
        # If it doesn't exist locally, try downloading it from S3
        local_path = download_from_s3(file_path)

    try:
        ext = os.path.splitext(local_path)[1].lower()
        if ext == ".pdf":
            text = load_pdf(local_path)
        elif ext == ".docx":
            text = load_docx(local_path)
        elif ext == ".txt":
            with open(local_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return text
    finally:
        # Clean up the temporary file if we downloaded it from S3
        if is_s3 and os.path.exists(local_path):
            os.remove(local_path)
