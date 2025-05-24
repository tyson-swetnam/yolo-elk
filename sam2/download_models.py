#!/usr/bin/env python3
"""
Download SAM2 model files
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def main():
    """Download SAM2 model files."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # SAM2 model URLs (these are the official model URLs)
    model_urls = {
        "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "sam2_hiera_l.yaml": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2_hiera_l.yaml"
    }
    
    print("Downloading SAM2 model files...")
    
    success_count = 0
    for filename, url in model_urls.items():
        filepath = models_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"✓ {filename} already exists")
            success_count += 1
            continue
            
        if download_file(url, filepath):
            success_count += 1
    
    print(f"\nDownload complete: {success_count}/{len(model_urls)} files")
    
    if success_count == len(model_urls):
        print("✓ All model files downloaded successfully!")
        
        # Update config to point to models directory
        print("\nUpdating configuration...")
        config_path = Path("configs/default_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Update paths to point to models directory
            config_content = config_content.replace(
                'model_checkpoint: "sam2_hiera_large.pt"',
                'model_checkpoint: "models/sam2_hiera_large.pt"'
            )
            config_content = config_content.replace(
                'model_config: "sam2_hiera_l.yaml"',
                'model_config: "models/sam2_hiera_l.yaml"'
            )
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            print("✓ Configuration updated")
        
    else:
        print("✗ Some downloads failed. Please check your internet connection.")

if __name__ == "__main__":
    main()
