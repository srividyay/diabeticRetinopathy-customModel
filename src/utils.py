import os, shutil, zipfile, io, csv

def safe_extract_zip(file_like: io.BytesIO, extract_to: str, max_files: int = 1000):
    with zipfile.ZipFile(file_like) as z:
        names = [n for n in z.namelist() if not n.endswith('/') and '__MACOSX' not in n and not os.path.basename(n).startswith('._')]
        if len(names) > max_files:
            names = names[:max_files]
        for n in names:
            dest = os.path.normpath(os.path.join(extract_to, os.path.basename(n)))
            with z.open(n) as src, open(dest, 'wb') as out:
                shutil.copyfileobj(src, out)

def write_csv(rows, out_path, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
