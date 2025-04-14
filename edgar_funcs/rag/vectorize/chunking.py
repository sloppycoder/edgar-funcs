import logging
import re

import html2text
import spacy
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# IMPORTANT:
# updaet the version if chunk size or chunking details are updated!
#
# the chunked text and their embeddings are store in pickle file for faster load
# update the chunking detail without version update can have unpredictable results
# when older and potentially obselete pickle files are loaded
CHUNK_ALORITHM_VERSION = "4"

DEFAULT_TEXT_CHUNK_SIZE = 1000

# version 3 uses chunk size of 3500


def chunk_text(
    content: str, chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE, method: str = "spacy"
) -> list[str]:
    if method == "spacy":
        return _chunk_text_by_spacy(content, chunk_size)
    else:
        raise RuntimeError(f"Unknown method: {method}")


# ruff: noqa: C901
def _chunk_text_by_spacy(content: str, chunk_size: int) -> list[str]:
    """
    Split a text into chunks of size chunk_size

    Args:
        content (str): The text to split into chunks
        chunk_size (int): The size of each chunk

    Returns:
        list[str]: A list of text chunks
    """

    # Load SpaCy NLP model
    nlp = spacy.load("en_core_web_sm")
    logger.debug("chunk_text: loaded model")

    chunks = []
    current_chunk = []
    current_size = 0

    # Split content into paragraphs (based on double newline)
    paragraphs = content.split("\n\n")

    for paragraph in paragraphs:
        # Detect potential tables by splitting into lines
        lines = paragraph.strip().split("\n")
        lines = [line.strip() for line in lines if not _is_line_empty(line)]

        # Buffer to collect a single Markdown table
        table_buffer = []

        for line in lines:
            is_table_row, is_empty_table_row = _check_table_row(line)
            if is_table_row:
                # Collect the line into the table buffer, only if the row is not empty
                if not is_empty_table_row:
                    table_buffer.append(line)
            else:
                # If line is not a table, flush the buffer as a table chunk
                if table_buffer:
                    table_content = "\n".join(table_buffer)
                    current_size = _add_to_chunk(
                        table_content, current_chunk, current_size, chunks, chunk_size
                    )
                    table_buffer = []  # Clear the buffer for the next table

                # Process non-table lines
                doc = nlp(line)
                sentences = [sent.text for sent in doc.sents]
                for sentence in sentences:
                    current_size = _add_to_chunk(
                        sentence, current_chunk, current_size, chunks, chunk_size
                    )

        # Flush any remaining table in the buffer
        if table_buffer:
            table_content = "\n".join(table_buffer)
            current_size = _add_to_chunk(
                table_content, current_chunk, current_size, chunks, chunk_size
            )

    # Add any remaining content
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Remove empty chunks
    return [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 100]


def trim_html_content(content: str) -> str:
    """
    remove the hidden div and convert the rest of html into text
    """
    if not content:
        return ""

    soup = BeautifulSoup(content, "html.parser")

    style_lambda = lambda value: value and "display:none" in value.replace(" ", "")  # noqa
    div_to_remove = soup.find("div", style=style_lambda)

    if div_to_remove:
        div_to_remove.decompose()  # type: ignore

    return _text2html(str(soup))


def _text2html(html_content: str):
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_emphasis = True
    converter.body_width = 0
    return converter.handle(html_content)


def _add_to_chunk(
    content_piece: str,
    current_chunk: list[str],
    current_size: int,
    chunks: list[str],
    chunk_size: int,
) -> int:
    """
    Add a piece of content to the current chunk or start a new one if the size exceeds
    chunk_size.

    Args:
        content_piece (str): The content to add.
        current_chunk (list[str]): The current chunk being built.
        current_size (int): The size of the current chunk.
        chunks (list[str]): The list of completed chunks.

    Returns:
        int: The updated size of the current chunk.
    """
    content_size = len(content_piece)
    if current_size + content_size > chunk_size:
        # Save current chunk and start a new one
        chunks.append("\n\n".join(current_chunk))
        current_chunk[:] = [content_piece]  # Reset current_chunk with new content
        return content_size
    else:
        current_chunk.append(content_piece)
        return current_size + content_size


def _is_line_empty(line: str) -> bool:
    content = line.strip()
    if not content:
        return True

    if len(content) < 5 and not any(char.isalpha() for char in content):
        return True

    words = re.findall(r"\b\w+\b", content)
    if all(len(word) <= 2 for word in words):
        return True

    return False


def _check_table_row(line: str) -> tuple[bool, bool]:
    """
    Check if a line is a table row in a Markdown table
    And if the table row is empty
    return [is_table_row, is_table_row_empty]
    """
    parts = [cell.strip() for cell in line.strip().split("|")]

    if len(parts) < 3:
        return False, False

    cells = [cell.strip() for cell in parts if cell.strip()]
    is_cell_empty = any(
        not s.strip() and bool(re.fullmatch(r"-*", s.strip())) for s in cells
    )

    return True, is_cell_empty
