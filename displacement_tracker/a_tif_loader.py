import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import click
import yaml

from displacement_tracker.util.env_loader import require_env_file


def download_tif_files_from_public_folder(search_string: str, download_dir="."):
    """
    Download all .tif files containing search_string from a public Google Drive folder using Google API.
    Args:
        folder_id (str): The ID of the public Google Drive folder.
        search_string (str): String to search for in file names.
        download_dir (str): Directory to save downloaded files.
    """
    folder_id = os.getenv("GDRIVE_ID")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = input("Enter your Google API key: ").strip()
        except EOFError:
            print("No API key provided. Exiting.")
            return
        if not api_key:
            print("No API key provided. Exiting.")
            return
    service = build("drive", "v3", developerKey=api_key)

    # Recursively search for .tif files in the folder and all subfolders
    def get_all_tif_files(parent_id):
        tif_files = []
        # List .tif files in this folder
        query = f"'{parent_id}' in parents and mimeType='image/tiff' and trashed=false"
        page_token = None
        while True:
            results = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            files = results.get("files", [])
            tif_files.extend(files)
            page_token = results.get("nextPageToken", None)
            if not page_token:
                break
        # List subfolders
        folder_query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        page_token = None
        while True:
            results = (
                service.files()
                .list(
                    q=folder_query,
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            folders = results.get("files", [])
            for folder in folders:
                tif_files.extend(get_all_tif_files(folder["id"]))
            page_token = results.get("nextPageToken", None)
            if not page_token:
                break
        return tif_files

    all_tif_files = get_all_tif_files(folder_id)
    matched_files = [f for f in all_tif_files if search_string in f["name"]]
    if not matched_files:
        print(
            f"No .tif files containing '{search_string}' found in folder or subfolders."
        )
        return
    os.makedirs(download_dir, exist_ok=True)
    for file in matched_files:
        file_id = file["id"]
        file_name = file["name"]
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(download_dir, file_name), "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"Downloading {file_name}...", end="\r")
        while not done:
            status, done = downloader.next_chunk()
        print(f"Downloaded {file_name}")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--download-dir",
    default=None,
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Directory to save files (overrides config)",
)
@require_env_file(["GOOGLE_API_KEY", "GDRIVE_ID"])
def tif_loader(config: str, download_dir: str | None) -> None:
    """
    Download .tif files from a public Google Drive folder for each search string in a YAML config file.
    The YAML file must have a section:
    loading:
      files:
        - search_string_1
        - search_string_2
    Optionally, you can specify download_dir in the config or via --download-dir.
    """
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    search_strings = cfg.get("loading", {}).get("files", [])
    if not search_strings:
        print("No search strings found in config under loading/files.")
        return
    # Determine download directory
    config_dir = cfg.get("geotiff_dir", ".")
    out_dir = download_dir if download_dir is not None else config_dir
    for search_string in search_strings:
        print(f"\n=== Downloading for search string: {search_string} ===")
        download_tif_files_from_public_folder(search_string, out_dir)


if __name__ == "__main__":
    tif_loader()
