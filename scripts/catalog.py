import io
import pickle  # Added import for StringIO

import pandas as pd

from edgar_funcs.edgar import edgar_file


def load_master_idx(
    start_year: int, end_year: int, filing_types: list[str]
) -> pd.DataFrame | None:
    all_df = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if year == 2025 and quarter > 1:
                continue

            master_idx_filename = f"edgar/full-index/{year}/QTR{quarter}/master.idx"
            print(f"Downloading {master_idx_filename}")
            content = edgar_file(master_idx_filename)
            idx_df = pd.read_csv(
                io.StringIO(content),
                sep="|",
                skiprows=11,
                names=["cik", "company_name", "form_type", "date_filed", "idx_filename"],
                dtype=str,
            )
            idx_df = idx_df[idx_df["form_type"].isin(filing_types)]
            idx_df["accession_number"] = idx_df["idx_filename"].apply(
                lambda x: x.split("/")[-1].split(".")[0]
            )
            all_df.append(idx_df)

    if all_df:
        return pd.concat(all_df)

    return None


def gen_catalog(output_file: str, start_year: int, end_year: int):
    filing_types = ["485BPOS"]
    idx_df = load_master_idx(start_year, end_year, filing_types)

    if idx_df is not None and len(idx_df) > 0:
        with open(output_file, "wb") as f:
            pickle.dump(idx_df, f)
            print(f"saved {len(idx_df)} rows to {output_file}")
    else:
        print("No data found for the specified filing types.")


if __name__ == "__main__":
    gen_catalog("all_485bpos_pd.pickle", 2008, 2024)
