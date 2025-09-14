import os
import re
import io
import time
import json
import hashlib
from pathlib import Path
import requests
import streamlit as st

# =============================================================
# Google Drive cached downloader
#   - Works WITHOUT a Google service account (public file ID)
#   - Also supports Drive API when service_account.json is present
#   - Local cache: only downloads once; reuses existing file
#   - "Force refresh" button to clear cache and re-download
# =============================================================

_DEF_CHUNK = 1 << 20  # 1 MiB

# ---------------------------
# Utility: hashing & writing
# ---------------------------

def _sha256_file(path: str, chunk_size: int = _DEF_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_to_file(resp: requests.Response, dst_path: str, chunk_size: int = _DEF_CHUNK):
    resp.raise_for_status()
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{dst_path}.part"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dst_path)


# -----------------------------------------
# Google Drive (PUBLIC) download, cached
# -----------------------------------------

def _gdrive_confirm_token(resp: requests.Response) -> str | None:
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    if "confirm=" in resp.url:
        return resp.url.split("confirm=")[1].split("&")[0]
    return None


def _extract_file_id(maybe_id_or_url: str) -> str:
    """Accepts a raw file ID or a Google Drive URL and returns the file ID."""
    s = maybe_id_or_url.strip()
    # Looks like a raw ID
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    # Patterns from typical Drive links
    m = re.search(r"/file/d/([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    raise ValueError("Could not extract Google Drive file ID from input. Provide a file ID or a valid Drive URL.")


def download_gdrive_public_cached(file_id_or_url: str, dst_path: str, expected_sha256: str | None = None, timeout: int = 60) -> str:
    """
    Download a **public** Google Drive file (no GCP account) exactly once and reuse local cache.
    Uses the confirm-token flow for large files. Optionally verifies SHA-256.
    """
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        if expected_sha256:
            try:
                if _sha256_file(dst_path) == expected_sha256.lower():
                    return dst_path
            except Exception:
                pass
        else:
            return dst_path

    file_id = _extract_file_id(file_id_or_url)
    sess = requests.Session()
    base = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    r = sess.get(base, params=params, stream=True, timeout=timeout)
    token = _gdrive_confirm_token(r)
    if token:
        params["confirm"] = token
        r = sess.get(base, params=params, stream=True, timeout=timeout)

    # If Drive sends an HTML error page (quota/permission), detect and raise
    ctype = r.headers.get("Content-Type", "").lower()
    if "text/html" in ctype:
        head = r.text[:1024].lower()
        if any(x in head for x in ("quota", "access denied", "sign in", "account")):
            raise RuntimeError("Google Drive public link error: quota or permission issue.")

    _stream_to_file(r, dst_path)

    if expected_sha256 and _sha256_file(dst_path) != expected_sha256.lower():
        raise RuntimeError("SHA-256 mismatch after download.")
    return dst_path


# -----------------------------------------
# Google Drive API (Service Account) cached
#  - Imported lazily so environments without
#    google libs will still run public-mode
# -----------------------------------------

class _DriveApiUnavailable(Exception):
    pass


def ensure_cached_gdrive_download_api(sa_json_path: str, file_id: str, dst_path: str,
                                      expected_sha256: str | None = None,
                                      verify_md5_with_drive: bool = True,
                                      copy_on_quota: bool = True,
                                      chunk_size: int = _DEF_CHUNK,
                                      max_retries: int = 5) -> str:
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError
    except Exception as e:
        raise _DriveApiUnavailable(f"Google Drive API libs not available: {e}")

    creds = Credentials.from_service_account_file(sa_json_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _is_quota_error(err: HttpError) -> bool:
        try:
            data = json.loads(err.content.decode("utf-8"))
            for e in data.get("error", {}).get("errors", []):
                if e.get("reason") in {"downloadQuotaExceeded", "rateLimitExceeded"}:
                    return True
        except Exception:
            pass
        return err.resp.status in (403, 429)

    def _md5_file(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    # Cache check with optional MD5 from Drive metadata
    p = Path(dst_path)
    if p.exists() and p.stat().st_size > 0 and verify_md5_with_drive:
        try:
            meta = svc.files().get(fileId=file_id, fields="md5Checksum", supportsAllDrives=True).execute()
            drive_md5 = meta.get("md5Checksum")
            if drive_md5:
                if _md5_file(dst_path) == drive_md5:
                    if expected_sha256:
                        assert _sha256_file(dst_path) == expected_sha256.lower(), "SHA-256 mismatch with expectation"
                    return dst_path
        except Exception:
            # If metadata fails, fall through to trust local or re-download below
            pass
    elif p.exists() and p.stat().st_size > 0 and not verify_md5_with_drive:
        if expected_sha256:
            try:
                if _sha256_file(dst_path) == expected_sha256.lower():
                    return dst_path
            except Exception:
                pass
        else:
            return dst_path

    # Download via API
    req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dst_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req, chunksize=chunk_size)
        attempt = 0
        while True:
            try:
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                break
            except HttpError as e:
                attempt += 1
                if _is_quota_error(e):
                    if copy_on_quota:
                        # Make a fresh copy and retry once
                        new_id = svc.files().copy(fileId=file_id, body={"name": f"copy_{int(time.time())}"}, fields="id", supportsAllDrives=True).execute()["id"]
                        req = svc.files().get_media(fileId=new_id, supportsAllDrives=True)
                        continue
                    raise
                if attempt > max_retries:
                    raise
                time.sleep(2 ** attempt)

    if expected_sha256 and _sha256_file(dst_path) != expected_sha256.lower():
        raise RuntimeError("SHA-256 mismatch after download.")
    return dst_path


# -----------------------------------------
# High-level helper: choose API or PUBLIC
# -----------------------------------------

def prepare_model_from_gdrive(file_id_or_url: str, dst_path: str,
                              sa_json_path: str | None = None,
                              expected_sha256: str | None = None,
                              force_public: bool = False) -> tuple[str, str]:
    """
    Returns (local_path, method_used) where method_used is "Drive API" or "Public link".
    If a service account JSON path is provided and libs are available, uses API unless
    force_public=True.
    """
    if not force_public and sa_json_path and os.path.exists(sa_json_path):
        try:
            file_id = _extract_file_id(file_id_or_url)
            out = ensure_cached_gdrive_download_api(sa_json_path, file_id, dst_path, expected_sha256=expected_sha256)
            return out, "Drive API"
        except _DriveApiUnavailable:
            pass  # fall back to public
        except Exception as e:
            # If API path fails (e.g., not shared with SA), fall back to public
            # The public path will still fail if the file is not publicly shared
            pass
    out = download_gdrive_public_cached(file_id_or_url, dst_path, expected_sha256=expected_sha256)
    return out, "Public link"


# =====================
# Streamlit integration
# =====================
# Env/secrets configuration (all optional except FILE_ID/URL)
GDRIVE_FILE_ID_OR_URL = os.getenv("GDRIVE_FILE_ID", "1O5TnYOzuZT2_nsG3EnWqAQDrWq4Pk5hH")  # can be a raw ID or a full URL
#LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "artifacts/model.bin")
LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "artifacts")
SA_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")  # path to service_account.json if available
EXPECTED_SHA256 = os.getenv("MODEL_SHA256", "").lower() or None

st.title("Model Preloader (Google Drive, cached)")

colA, colB, colC = st.columns([1,1,1])
with colA:
    force_public = st.checkbox("Force public method", value=not bool(SA_JSON and os.path.exists(SA_JSON)))
with colB:
    clear_cache = st.button("Force refresh (clear cache)")
with colC:
    show_info = st.checkbox("Show debug info", value=False)

if clear_cache and os.path.exists(LOCAL_PATH):
    try:
        os.remove(LOCAL_PATH)
        st.info("Local cache cleared.")
    except Exception as e:
        st.warning(f"Could not clear cache: {e}")

if not GDRIVE_FILE_ID_OR_URL:
    st.error("GDRIVE_FILE_ID is not set. Provide a Google Drive file ID or URL via env var.")
    st.stop()

with st.spinner("Preparing model (cached)…"):
    try:
        Path(LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
        local_path, method = prepare_model_from_gdrive(
            file_id_or_url=GDRIVE_FILE_ID_OR_URL,
            dst_path=LOCAL_PATH,
            sa_json_path=SA_JSON if SA_JSON else None,
            expected_sha256=EXPECTED_SHA256,
            force_public=force_public,
        )
        st.success(f"Model ready: {local_path}  •  via {method}")
        if show_info:
            st.caption(f"SHA-256: {_sha256_file(local_path)}")
    except Exception as e:
        st.error(
            "Download failed.\n\n"
            "If using Drive API: ensure service_account.json exists and the file/folder is shared with that SA email.\n"
            "If using Public link: ensure the file is shared 'Anyone with the link'. Large/public files can hit quotas.\n\n"
            f"Technical detail: {e}"
        )
        st.stop()

