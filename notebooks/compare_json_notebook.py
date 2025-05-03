import marimo

__generated_with = "0.13.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from dotenv import load_dotenv
    from utils import prep_compare_df

    load_dotenv()

    _columns_to_select = " batch_id, cik, accession_number, extraction_type, response, selected_chunks, model"

    # batch_id for trustees
    # _old_batch_id = "20250422004812-oxk"
    # _new_batch_id = "20250503013026-bbb"

    # batch_id for fundmgr
    _old_batch_id = "20250422002102-ynv"
    _new_batch_id = "20250503010430-nlu"

    df_compare = prep_compare_df(_old_batch_id, _new_batch_id)
    print(f"found {len(df_compare)} different rows")

    mo.md("Loading data from BigQuery")

    return df_compare, mo


@app.cell(hide_code=True)
def _(df_compare, mo):
    compare_list = []
    for index, row in df_compare.iterrows():
        compare_dict = {
            "filing": f"{row['cik']}/{row['accession_number']}",
            "entities": f"{str(row['n_entity_old'])} - {str(row['n_entity_new'])}",
            "chunks": f"{row['chunks_old']} vs {row['chunks_new']}",
            "index": index,
        }
        compare_list.append(compare_dict)

    compare_table = mo.ui.table(
        data=compare_list, pagination=True, selection="single", page_size=5
    )
    compare_table
    return (compare_table,)


@app.cell(hide_code=True)
def _(compare_table, df_compare, mo):
    _old_stack = mo.vstack([mo.md("old:"), mo.ui.text_area("")])
    _new_stack = mo.vstack([mo.md("new:"), mo.ui.text_area("")])

    if compare_table.value:
        _filing = compare_table.value[0]["filing"]
        _selected_index = compare_table.value[0]["index"]

        _row = df_compare.loc[_selected_index]

        _response_old = df_compare.loc[_selected_index]["response_old"]
        _response_new = df_compare.loc[_selected_index]["response_new"]

        _text_area_old = mo.ui.text_area(_response_old)
        _text_area_new = mo.ui.text_area(_response_new)

        _old_stack = mo.vstack(
            [
                mo.md(f"old:{_row['n_entity_old']}"),
                mo.ui.text_area(_response_old, rows=25),
            ]
        )
        _new_stack = mo.vstack(
            [
                mo.md(f"new:{_row['n_entity_new']}"),
                mo.ui.text_area(_response_new, rows=25),
            ]
        )

    mo.hstack([_old_stack, _new_stack])

    return


if __name__ == "__main__":
    app.run()
