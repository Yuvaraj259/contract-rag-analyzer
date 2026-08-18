import os
import urllib.request
import sys

def download_model():
    model_name = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    url = f"https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/{model_name}"
    cache_dir = "model_cache/gpt4all"
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, model_name)
    
    if os.path.exists(file_path):
        print(f"Model already exists at {file_path}")
        return

    print(f"Downloading {model_name} (approx 4.7 GB). This will take several minutes...")
    
    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()
        
    try:
        urllib.request.urlretrieve(url, file_path, reporthook=report)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    download_model()
