import os
import urllib.request

os.makedirs("./weights", exist_ok=True)

files = {
    "blur_jpg_prob0.1.pth": "https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1",
    "blur_jpg_prob0.5.pth": "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1",
}

for filename, url in files.items():
    out_path = f"./weights/{filename}"
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to: {out_path}")