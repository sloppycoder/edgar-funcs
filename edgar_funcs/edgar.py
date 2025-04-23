import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Lee Lynn (hayashi@yahoo.com)"
EDGAR_BASE_URL = "https://www.sec.gov/Archives"

# Document tag contents usually looks like this,
# FILENAME and DESCRIPTION are optional
#
# <DOCUMENT>
# <TYPE>stuff
# <SEQUENCE>stuff
# <FILENAME>stuff
# <TEXT>
# Document 1 - file: stuff
#
# </DOCUMENT>
# the regex below tries to parse
# doc element in index-headers.html
_doc_regex = re.compile(
    r"""<DOCUMENT>\s*
<TYPE>(?P<type>.*?)\s*
<SEQUENCE>(?P<sequence>.*?)\s*
<FILENAME>(?P<filename>.*?)\s*
(?:<DESCRIPTION>(?P<description>.*?)\s*)?
<TEXT>
(?P<text>.*?)
</TEXT>""",
    re.DOTALL | re.VERBOSE | re.IGNORECASE,
)

# in SEC_HEADER
# FILED AS OF DATE:		20241017
_date_filed_regex = re.compile(r"FILED AS OF DATE:\s*(\d{8})", re.IGNORECASE)


class RateLimitedException(Exception):
    pass


class InvalidFilingExceptin(Exception):
    pass


class SECFiling:
    def __init__(
        self,
        cik: str = "",
        accession_number: str = "",
        idx_filename: str = "",
        prefer_index_headers: bool = True,  # debug only, always set to True in production
    ) -> None:
        # sometimes a same filename is used by several CIKs
        # filename as in master.idx
        # e.g. edgar/data/106830/0001683863-20-000050.txt
        # if filename is specified, we derive cik and accession_number from it
        if idx_filename:
            self.idx_filename = idx_filename
            self.cik, self.accession_number = parse_idx_filename(idx_filename)
        else:
            if cik and accession_number:
                self.cik, self.accession_number = cik, accession_number
                self.idx_filename = f"edgar/data/{cik}/{accession_number}.txt"
            else:
                raise ValueError(
                    "cik and accession_number must be specified when idx_filename is not"
                )

        # idx filename for the filing index-headers file
        self.index_html_path = _index_html_path(self.idx_filename)
        self.index_headers_path = self.index_html_path.replace(
            "-index.html", "-index-headers.html"
        )
        self.date_filed = ""
        self.documents = []

        if prefer_index_headers:
            (self.date_filed, self.documents) = self._read_index_headers()

        if not self.documents:
            # for older filings index-headers.htm l does not exist
            # so parsing the index.html instead
            (self.date_filed, self.documents) = self._read_index()

        logger.debug(f"initialized SECFiling({self.cik},{self.idx_filename})")

    def get_doc_path(self, doc_type: str) -> list[str]:
        """
        Reads the contents of documents of a specific type from the filing.

        Args:
            doc_type (str): The type of document to read (e.g., "485BPOS").

        Returns:
            list[str]: A list of filenames that matches the doc_type

        Raises:
            InvalidFilingExceptin: If the specified document type is not found in the
            filing or if the document path cannot be determined.
        """
        # Get the paths of documents of the specified type
        paths = [doc["filename"] for doc in self.documents if doc["type"] == doc_type]
        if paths is None or paths == []:
            raise InvalidFilingExceptin(
                f"{self.idx_filename} does not contain a {doc_type} document"
            )
        return [str(Path(self.index_headers_path).parent / path) for path in paths]

    def get_doc_content(
        self, doc_type: str, file_types: list[str]
    ) -> list[tuple[str, str]]:
        result = []
        for doc_path in self.get_doc_path(doc_type):
            doc_type = doc_path.split(".")[-1]
            if doc_type not in file_types:
                continue
            content = edgar_file(doc_path)
            if content:
                result.append((doc_path, content))
        return result

    def _read_index_headers(self) -> tuple[str, list[dict[str, Any]]]:
        """read the index-headers.html file and extract document list from sec_header"""

        content = edgar_file(self.index_headers_path)
        if not content:
            logger.debug(
                f"Unable to download {self.index_headers_path}, perhaps the filing is too old?"  # noqa E501
            )
            return "", []

        soup = BeautifulSoup(content, "html.parser")
        # each index-headers.html file contains a single <pre> tag
        # inside there are SGML content of meta data for the filing
        pre = soup.find("pre")
        if pre is None:
            logger.debug(f"No <pre> tag found in {self.index_headers_path}")
            return "", []

        pre_soup = BeautifulSoup(pre.get_text(), "html.parser")

        sec_header = pre_soup.find("sec-header")
        sec_header_text = ""
        date_filed = ""
        if sec_header:
            sec_header_text = str(sec_header)
            match = _date_filed_regex.search(sec_header_text)
            if match:
                digits = match.group(1)
                date_filed = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"

            documents = []
            for doc in pre_soup.find_all("document"):
                match = _doc_regex.search(str(doc))

                if match:
                    doc_info = {
                        "type": match.group("type"),
                        "sequence": match.group("sequence"),
                        "filename": match.group("filename"),
                    }
                    documents.append(doc_info)

            return date_filed, documents

        logger.info(f"No sec-header found in {self.index_headers_path}")

        return "", []

    def _read_index(self) -> tuple[str, list[dict[str, Any]]]:
        """read the index.html file and extract the document list form html table"""

        content = edgar_file(self.index_html_path)
        if not content:
            logger.debug(f"Unable to download {self.index_html_path}")
            return "", []

        soup = BeautifulSoup(content, "html.parser")

        filing_date = ""
        info_heads = soup.find_all("div", class_="infoHead")
        for head in info_heads:
            if head.text.strip() == "Filing Date":
                filing_date = head.find_next_sibling("div", class_="info").text.strip()  # type: ignore
                break

        table = soup.find("table", class_="tableFile")

        documents = []
        for row in table.find_all("tr")[1:]:  # type: ignore Skip the header row
            cols = row.find_all("td")  # type: ignore
            if len(cols) < 5:
                continue

            sequence = cols[0].get_text(strip=True) or None
            filename = cols[2].find("a").text if cols[2].find("a") else None  # type: ignore
            doc_type = cols[3].get_text(strip=True) or None

            if filename:
                documents.append(
                    {"sequence": sequence, "filename": filename, "type": doc_type}
                )

        return filing_date, documents

    def __str__(self):
        return f"SECFiling({self.cik},{self.accession_number},{self.date_filed},docs={len(self.documents)})"  # noqa E501


def parse_idx_filename(idx_filename: str) -> tuple[str, str]:
    "Determine CIK and Accession Number from index filename"
    match = re.search(r"edgar/data/(\d+)/(.+)\.txt", idx_filename)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"parse_idx_filename: {idx_filename} is of an unexpected format")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(RateLimitedException),
)
def edgar_file(
    idx_filename: str,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str | None:
    """
    Download a file from EDGAR site and return its content as a string
    e.g. edgar/data/123456/0001234567-21-000123.txt
    """
    response = requests.get(
        f"{EDGAR_BASE_URL}/{idx_filename}", headers={"User-Agent": user_agent}
    )
    if response.status_code == 200:
        return response.text
    elif response.status_code == 429:
        # TODO: add retrying logic and etc to make it more robust
        logger.debug(f"Received 429 trying to download {idx_filename}")
        raise RateLimitedException(f"Rate limited: {idx_filename}")
    else:
        logger.info(f"Failed to download from {idx_filename}: {response.status_code}")


def _index_html_path(idx_filename: str) -> str:
    """
    convert a filename from master.idx filename to -index.html
    e.g.
    edgar/data/1007571/000119312524109215/0001193125-24-109215-index.html
    """
    filepath = Path(idx_filename)
    basename = filepath.name.replace(".txt", "")
    return str(filepath.parent / basename.replace("-", "") / f"{basename}-index.html")


@lru_cache(maxsize=1)
def load_filing_catalog(start_date: str, end_date: str) -> pd.DataFrame:
    """
    load local copy of 485BPOS filings catalog
    """
    data_path = Path(__file__).parent / "data/catalog"
    catalog_path = data_path / "all_485bpos_pd.pickle"
    df_filings = pd.read_pickle(catalog_path)
    assert len(df_filings) > 10000

    df_cik = pd.read_csv(
        data_path / "interested_cik_list.csv",
    )
    assert len(df_cik) > 1000

    # Filter rows by date range and by interested CIK list
    interested_ciks = df_cik["cik"].astype(str).tolist()
    df_result = df_filings[
        (df_filings["date_filed"] > start_date)
        & (df_filings["date_filed"] < end_date)
        & (df_filings["cik"].isin(interested_ciks))
    ]
    return df_result
